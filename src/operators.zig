const std = @import("std");
const Tensor = @import("tensor.zig");

pub fn rmsnorm(out: Tensor, x: Tensor) void {
    std.debug.print("{any}, {any}", .{ out, x });
}

pub fn softmax(x: Tensor) void {
    std.debug.print("{any}", .{x});
}

pub fn matmul(out: Tensor, x: Tensor, weight: Tensor) void {
    std.debug.print("{any}, {any}, {any}", .{ out, x, weight });
}

pub fn embedRope(x: Tensor) void {
    std.debug.print("{any}", .{x});
}
