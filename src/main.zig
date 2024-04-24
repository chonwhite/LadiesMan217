const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    const args = try std.process.argsAlloc(std.heap.page_allocator);
    defer std.process.argsFree(std.heap.page_allocator, args);

    for (args, 0..) |arg, i| {
        try stdout.print("arg {}: {s}\n", .{ i, arg });
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    // get an std.mem.Allocator from it
    const allocator = gpa.allocator();
    const path: []u8 = args[1];
    std.debug.print("Allocator:{any}, path:{any}", .{ allocator, path });

    const transformer = @import("transformer.zig").loadModel(allocator, path);
    std.debug.print("transformer:{any}", .{transformer});
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
