//! Scarlet: native MemoryMap/MemoryUnmap/MemoryProtect system calls on the
//! file's native handle (`std::os::fd::AsRawFd` returns it).
//!
//! The native ABI returns the mapped address or `usize::MAX`; there is no
//! errno, so failures are reported as `ErrorKind::Other`. Scarlet has no
//! msync, so flushing a writable shared mapping is unsupported.
//!
//! Read-only maps use private mappings. A shared file mapping copies the
//! whole file into kernel memory and writes it back when unmapped, even if
//! nothing was written; private mappings fault pages in from the file, and a
//! read-only mapping cannot tell the two apart.
//!
//! Some filesystems cannot map files at all (the initramfs, and so the lower
//! layer of an overlay root). A private mapping is then filled by reading the
//! file into anonymous memory: it behaves the same, except that it never sees
//! later changes to the file, which a private mapping need not either. The
//! file's cursor is restored afterwards, as mapping never moves it. Shared
//! mappings have no such fallback.
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;

const MEMORY_MAP: usize = 700;
const MEMORY_UNMAP: usize = 701;
const MEMORY_PROTECT: usize = 702;

const PROT_READ: usize = 0x1;
const PROT_WRITE: usize = 0x2;
const PROT_EXEC: usize = 0x4;
const MAP_SHARED: usize = 0x01;
const MAP_PRIVATE: usize = 0x02;
const MAP_ANONYMOUS: usize = 0x20;

/// Scarlet maps 4 KiB pages on every supported architecture.
const PAGE_SIZE: usize = 0x1000;
const FAILED: usize = usize::MAX;

/// Issues a native system call with up to six arguments.
///
/// # Safety
///
/// The arguments must be valid for `number`.
unsafe fn syscall(number: usize, args: [usize; 6]) -> usize {
    let result;
    // SAFETY: the caller upholds the call's contract. The kernel may write
    // memory through the arguments, so this is not `nomem`/`readonly`.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!(
            "svc #0",
            in("x8") number,
            inlateout("x0") args[0] => result,
            in("x1") args[1],
            in("x2") args[2],
            in("x3") args[3],
            in("x4") args[4],
            in("x5") args[5],
            clobber_abi("C"),
            options(nostack),
        );
        #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
        core::arch::asm!(
            "ecall",
            in("a7") number,
            inlateout("a0") args[0] => result,
            in("a1") args[1],
            in("a2") args[2],
            in("a3") args[3],
            in("a4") args[4],
            in("a5") args[5],
            clobber_abi("C"),
            options(nostack),
        );
        #[cfg(not(any(
            target_arch = "aarch64",
            target_arch = "riscv32",
            target_arch = "riscv64"
        )))]
        compile_error!("Scarlet memmap2 supports AArch64 and RISC-V targets only");
    }
    result
}

fn failed(operation: &str) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("Scarlet {operation} failed"))
}

pub struct MmapInner {
    ptr: *mut u8,
    len: usize,
}

impl MmapInner {
    /// Maps `len` bytes of `handle` (or anonymous memory) at `offset`.
    fn new(len: usize, prot: usize, flags: usize, handle: usize, offset: u64) -> io::Result<Self> {
        if std::mem::size_of::<usize>() < 8 && len > isize::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "memory map length overflows isize",
            ));
        }
        let alignment = (offset % PAGE_SIZE as u64) as usize;
        let aligned_offset = usize::try_from(offset - alignment as u64)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset out of range"))?;
        // Zero-length mappings are refused by the kernel; map one byte as
        // other platforms do, while the public length stays zero.
        let map_len = (len + alignment).max(1);
        // SAFETY: a null address hint never replaces an existing mapping.
        let address = unsafe {
            syscall(
                MEMORY_MAP,
                [handle, 0, map_len, prot, flags, aligned_offset],
            )
        };
        if address == FAILED || address % PAGE_SIZE != 0 {
            return Err(failed("MemoryMap"));
        }
        Ok(Self {
            ptr: (address + alignment) as *mut u8,
            len,
        })
    }

    fn file(len: usize, file: &File, offset: u64, prot: usize, flags: usize) -> io::Result<Self> {
        let handle = usize::try_from(file.as_raw_fd())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid file handle"))?;
        match Self::new(len, prot, flags, handle, offset) {
            Err(error) if flags & MAP_PRIVATE != 0 => {
                Self::copied(len, file, offset, prot).map_err(|_| error)
            }
            result => result,
        }
    }

    /// A private mapping filled from the file, for files that cannot be mapped.
    /// Bytes past the end of the file read as zero.
    fn copied(len: usize, file: &File, offset: u64, prot: usize) -> io::Result<Self> {
        let mut map = Self::new(
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            0,
            0,
        )?;
        let mut reader = file;
        let cursor = reader.stream_position()?;
        reader.seek(SeekFrom::Start(offset))?;
        // SAFETY: the anonymous mapping is writable and at least `len` bytes.
        let mut buffer = unsafe { std::slice::from_raw_parts_mut(map.ptr, len) };
        let filled = loop {
            if buffer.is_empty() {
                break Ok(());
            }
            match reader.read(buffer) {
                Ok(0) => break Ok(()),
                Ok(n) => buffer = &mut buffer[n..],
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => break Err(e),
            }
        };
        reader.seek(SeekFrom::Start(cursor))?;
        filled?;
        if prot & PROT_WRITE == 0 {
            map.protect(prot)?;
        }
        Ok(map)
    }

    /// The page-aligned start and length of the whole mapping.
    fn mapping(&self) -> (usize, usize) {
        let alignment = self.ptr as usize % PAGE_SIZE;
        (self.ptr as usize - alignment, (self.len + alignment).max(1))
    }

    pub fn map(
        len: usize,
        file: &File,
        offset: u64,
        _: bool,
        _: bool,
        _: bool,
    ) -> io::Result<Self> {
        Self::file(len, file, offset, PROT_READ, MAP_PRIVATE)
    }

    pub fn map_exec(
        len: usize,
        file: &File,
        offset: u64,
        _: bool,
        _: bool,
        _: bool,
    ) -> io::Result<Self> {
        Self::file(len, file, offset, PROT_READ | PROT_EXEC, MAP_PRIVATE)
    }

    pub fn map_mut(
        len: usize,
        file: &File,
        offset: u64,
        _: bool,
        _: bool,
        _: bool,
    ) -> io::Result<Self> {
        Self::file(len, file, offset, PROT_READ | PROT_WRITE, MAP_SHARED)
    }

    pub fn map_copy(
        len: usize,
        file: &File,
        offset: u64,
        _: bool,
        _: bool,
        _: bool,
    ) -> io::Result<Self> {
        Self::file(len, file, offset, PROT_READ | PROT_WRITE, MAP_PRIVATE)
    }

    pub fn map_copy_read_only(
        len: usize,
        file: &File,
        offset: u64,
        _: bool,
        _: bool,
        _: bool,
    ) -> io::Result<Self> {
        Self::file(len, file, offset, PROT_READ, MAP_PRIVATE)
    }

    pub fn map_anon(
        len: usize,
        _stack: bool,
        _populate: bool,
        _huge: Option<u8>,
        _no_reserve: bool,
    ) -> io::Result<Self> {
        Self::new(
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            0,
            0,
        )
    }

    pub fn flush(&self, _: usize, _: usize) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "flushing a memory map is not supported on Scarlet",
        ))
    }

    pub fn flush_async(&self, offset: usize, len: usize) -> io::Result<()> {
        self.flush(offset, len)
    }

    fn protect(&mut self, prot: usize) -> io::Result<()> {
        let (address, len) = self.mapping();
        // SAFETY: the range is exactly this mapping.
        if unsafe { syscall(MEMORY_PROTECT, [address, len, prot, 0, 0, 0]) } == FAILED {
            return Err(failed("MemoryProtect"));
        }
        Ok(())
    }

    pub fn make_read_only(&mut self) -> io::Result<()> {
        self.protect(PROT_READ)
    }

    pub fn make_exec(&mut self) -> io::Result<()> {
        self.protect(PROT_READ | PROT_EXEC)
    }

    pub fn make_mut(&mut self) -> io::Result<()> {
        self.protect(PROT_READ | PROT_WRITE)
    }

    #[inline]
    pub fn ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(clippy::unnecessary_wraps)]
    pub fn check_safe_to_map(
        _file: &File,
        _offset: u64,
        _len: usize,
    ) -> Result<(), crate::MapIfSafeError> {
        Err(crate::MapIfSafeError::new_not_supported())
    }
}

impl Drop for MmapInner {
    fn drop(&mut self) {
        let (address, len) = self.mapping();
        // Errors cannot be reported from Drop.
        // SAFETY: the range is exactly this mapping, which is no longer used.
        unsafe { syscall(MEMORY_UNMAP, [address, len, 0, 0, 0, 0]) };
    }
}

// SAFETY: the mapping is plain memory owned by this value, like on Unix.
unsafe impl Sync for MmapInner {}
unsafe impl Send for MmapInner {}

pub fn file_len(file: &File) -> io::Result<u64> {
    Ok(file.metadata()?.len())
}
