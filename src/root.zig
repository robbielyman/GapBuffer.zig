const gb = @import("gap_buffer.zig");
pub const GapBuffer = gb.GapBuffer;
pub const GapBufferAligned = gb.GapBufferAligned;
pub const GapBufferUnmanaged = gb.GapBufferUnmanaged;
pub const GapBufferAlignedUnmanaged = gb.GapBufferAlignedUnmanaged;

const mgb = @import("multi_gap_buffer.zig");
pub const MultiGapBuffer = mgb.MultiGapBuffer;

test "refAllDecls" {
    @import("std").testing.refAllDecls(@This());
}
