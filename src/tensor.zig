const std = @import("std");
const debug = std.debug;
const assert = debug.assert;
// const testing = std.testing;
const mem = std.mem;
const test_print = std.debug.print;

const Allocator = mem.Allocator;

pub fn Tensor(comptime T: type, comptime D: u8) type {
    return struct {
        const Self = @This();

        data: Slice,
        allocator: Allocator,
        dimensions: [D]u32,
        shape: [D]u32,

        pub const Slice = []T;

        pub fn init(allocator: Allocator, shape: [D]u32) !Self {
            var dimensions: [D]u32 = .{1} ** D;
            for (0..D) |i| {
                var dim: u32 = 1;
                var count = D - 1;
                while (count > i) {
                    test_print("\n i: {} count: {}", .{ i, count });
                    dim = dim * shape[count];
                    count -= 1;
                }
                dimensions[i] = dim;
            }

            var num: usize = 1;
            for (shape) |s| {
                num = num * s;
            }

            test_print("num: {}\n", .{num});
            const memory = try allocator.alloc(T, num);

            return Self{
                .data = memory,
                .allocator = allocator,
                .shape = shape,
                .dimensions = dimensions,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
        }

        pub inline fn get(self: Self, indices: [D]u8) T {
            var index: usize = 1;
            for (self.dimensions, 0..) |dim, i| {
                index += indices[i] * dim;
            }
            const val: T = self.data[index];
            test_print("\n indices: {any} = index: {any}, val = {any}", .{ indices, index, val });
            return val;
        }

        pub fn copyFrom(self: Self, other: Self) void {
            @memcpy(self.data, other.data);
        }

        pub fn print(self: *const Self) void {
            test_print("\n shape: {any}, dimensions: {any}", .{ self.shape, self.dimensions });
        }
    };
}

test "init" {
    test_print("xxx", .{});
    test_print("in test", .{});
    const tensor = try Tensor(f32, 4).init(std.testing.allocator, .{ 4, 8, 3, 2 });
    const t2 = tensor;
    defer tensor.deinit();

    tensor.print();
    t2.print();

    _ = tensor.get(.{ 3, 2, 2, 1 });
    test_print("\n dimensions: {any}", .{tensor.dimensions});

    // assert(tensor.dimensions.len == 2);
}
