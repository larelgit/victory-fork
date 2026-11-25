//! Copies out of title memory with plain loads, recovering from a page
//! fault, instead of through the kernel.
//!
//! The title may unmap pages of a range between its recording and its copy
//! (tiled resources, page mappings, the runtime remapping a view), so
//! copies out of title memory went through the kernel (`ReadProcessMemory`),
//! which reports such a page instead of faulting. That is a system call a
//! copy, most of the cost of the small ones: about 1 µs over a plain copy
//! of 4 KiB on a Ryzen 7 7800X3D, for tens of thousands of copies a frame
//! in gameplay.
//!
//! Here a copy is one `rep movsb` in a routine of its own, and a vectored
//! handler, first in line, takes a fault at that one instruction and
//! resumes after it, where the count left says the copy did not finish.
//! The caller then copies the whole range again through the kernel, so
//! what an unmapped page does to the result stays what it was.
//! `VICTORY_USER_COPIES=0` copies through the kernel only, as before.
//!
//! Copies into title memory stay with the kernel. Its pages are
//! write-watched, and each asking resets the watch, after which the first
//! store to every page faults; the kernel's stores take those faults for
//! less: a 64 KiB piece into freshly reset pages took 11.4 µs through
//! `NtWriteVirtualMemory` and 13.1 µs with a plain copy on the same
//! machine. A gameplay run with plain stores into title memory as well ran
//! at 188 ms a frame, against 178 and 181 with plain loads only.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(
    ".text",
    ".p2align 4",
    ".globl victory_user_copy",
    "victory_user_copy:",
    // `rdi` and `rsi` are the caller's: kept in `r9` and `r10`, which are
    // not, so the routine leaves the stack alone (a stack walk finds its
    // caller at `[rsp]`, as for any leaf).
    "mov r9, rdi",
    "mov r10, rsi",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    ".globl victory_user_copy_move",
    "victory_user_copy_move:",
    "rep movsb",
    ".globl victory_user_copy_after",
    "victory_user_copy_after:",
    "mov rax, rcx",
    "mov rdi, r9",
    "mov rsi, r10",
    "ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    /// Copies `len` bytes from `src` to `dst` (the direction flag is clear
    /// at a call); returns the bytes left when a fault stopped it, 0 when
    /// all were copied.
    fn victory_user_copy(dst: *mut u8, src: *const u8, len: usize) -> usize;
    /// The routine's `rep movsb`, and the instruction after it: labels, of
    /// which only the addresses are taken.
    fn victory_user_copy_move();
    fn victory_user_copy_after();
}

/// Whether copies try plain loads first (on unless `VICTORY_USER_COPIES=0`)
/// and the handler is in place.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(target_arch = "x86_64")
            && std::env::var_os("VICTORY_USER_COPIES").is_none_or(|v| v != "0")
            && install()
    })
}

/// Copies `len` bytes of title memory at `src` into host memory at `dst`
/// with plain loads and stores; false when that is off, the two overlap,
/// or a page of the source could not be read (then part of the range may
/// have been copied).
pub(crate) fn copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    // `rep movsb` copies forwards: ranges that overlap are left to the
    // kernel, as before.
    let (d, s) = (dst as usize, src as usize);
    if !on() || (d < s.saturating_add(len) && s < d.saturating_add(len)) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: a fault in the routine's copy is taken by `on_fault` and
        // ends the routine with bytes left; the caller vouches that `dst`
        // is host memory of `len` bytes that nothing else uses meanwhile.
        let left = unsafe { victory_user_copy(dst, src, len) };
        #[cfg(feature = "diagnostics")]
        if left != 0 {
            crate::copystat::add(crate::copystat::C::UserFaults, 1);
        }
        left == 0
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Registers the handler, first in line; false if Windows refused.
#[cfg(target_arch = "x86_64")]
fn install() -> bool {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_IN_PAGE_ERROR, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched and the
        // context it resumes the thread with.
        let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        if context.Rip != victory_user_copy_move as *const () as u64 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        match record.ExceptionCode {
            // A page not there or not readable (or, should the destination
            // be one, not writable), a mapped file's page that could not be
            // read, or a guard page (its guard is gone with the fault).
            // `rep movsb` stops at the byte it could not move, with `rcx`
            // the bytes left; the routine returns them.
            EXCEPTION_ACCESS_VIOLATION | EXCEPTION_IN_PAGE_ERROR | STATUS_GUARD_PAGE_VIOLATION => {
                context.Rip = victory_user_copy_after as *const () as u64;
                EXCEPTION_CONTINUE_EXECUTION
            }
            _ => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    // SAFETY: the handler only acts on faults of the routine above.
    !unsafe { AddVectoredExceptionHandler(1, Some(on_fault)) }.is_null()
}

#[cfg(not(target_arch = "x86_64"))]
fn install() -> bool {
    false
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS,
        PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const PAGE: usize = 4096;

    /// Four committed read-write pages, filled with a pattern; freed on drop.
    struct Pages(*mut u8);
    impl Pages {
        fn new() -> Self {
            // SAFETY: a fresh allocation of our own.
            let p = unsafe {
                VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            } as *mut u8;
            assert!(!p.is_null());
            for i in 0..4 * PAGE {
                // SAFETY: inside the allocation.
                unsafe { p.add(i).write((i * 7 + 3) as u8) };
            }
            Self(p)
        }
        fn protect(&self, page: usize, protect: u32) {
            let mut old = 0;
            // SAFETY: a page of our allocation.
            assert_ne!(unsafe { VirtualProtect(self.0.add(page * PAGE).cast(), PAGE, protect, &mut old) }, 0);
        }
    }
    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    fn expected(range: std::ops::Range<usize>) -> Vec<u8> {
        range.map(|i| (i * 7 + 3) as u8).collect()
    }

    #[test]
    fn copies_what_is_mapped() {
        let src = Pages::new();
        for (from, len) in [(0, 1), (13, 511), (100, 4096), (5, 4 * PAGE - 5)] {
            let mut dst = vec![0u8; len];
            assert!(copy(dst.as_mut_ptr(), src.0.wrapping_add(from), len));
            assert_eq!(dst, expected(from..from + len));
        }
        assert!(copy(std::ptr::null_mut(), std::ptr::null(), 0));
        // Overlapping ranges are not copied here.
        assert!(!copy(src.0.wrapping_add(8), src.0, 64));
    }

    #[test]
    fn stops_at_a_page_that_cannot_be_read() {
        let src = Pages::new();
        // A decommitted page, a page of no access, a guard page: each in
        // the middle of the copy.
        // SAFETY: the third page of our allocation.
        assert_ne!(unsafe { VirtualFree(src.0.add(2 * PAGE).cast(), PAGE, MEM_DECOMMIT) }, 0);
        src.protect(1, PAGE_NOACCESS);
        let mut dst = vec![0u8; 4 * PAGE];
        assert!(!copy(dst.as_mut_ptr(), src.0, 4 * PAGE));
        assert_eq!(dst[..PAGE], expected(0..PAGE)[..]);
        assert!(!copy(dst.as_mut_ptr(), src.0.wrapping_add(2 * PAGE + 8), 16));

        let guarded = Pages::new();
        guarded.protect(3, PAGE_READWRITE | PAGE_GUARD);
        let mut dst = vec![0u8; 2 * PAGE];
        assert!(!copy(dst.as_mut_ptr(), guarded.0.wrapping_add(2 * PAGE), 2 * PAGE));
        assert_eq!(dst[..PAGE], expected(2 * PAGE..3 * PAGE)[..]);
        // The guard went with the fault: the page reads now.
        assert!(copy(dst.as_mut_ptr(), guarded.0.wrapping_add(2 * PAGE), 2 * PAGE));
        assert_eq!(dst, expected(2 * PAGE..4 * PAGE));
    }

    #[test]
    fn stops_at_reserved_memory_and_at_a_destination_that_cannot_be_written() {
        // SAFETY: a fresh reservation, never committed; freed below.
        let reserved = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) } as *mut u8;
        assert!(!reserved.is_null());
        let mut dst = vec![0u8; 64];
        assert!(!copy(dst.as_mut_ptr(), reserved, 64));
        let src = Pages::new();
        assert!(!copy(reserved, src.0, 64));
        // SAFETY: our reservation.
        unsafe { VirtualFree(reserved.cast(), 0, MEM_RELEASE) };

        let readonly = Pages::new();
        readonly.protect(0, PAGE_READONLY);
        assert!(!copy(readonly.0, src.0.wrapping_add(PAGE), 64));
        let mut back = vec![0u8; 64];
        assert!(copy(back.as_mut_ptr(), readonly.0, 64));
        assert_eq!(back, expected(0..64));
    }
}
