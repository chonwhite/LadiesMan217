const std = @import("std");
const debug = std.debug;
const assert = debug.assert;
const mem = std.mem;
const test_print = std.debug.print;

const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;

const Tensor = @import("tensor.zig").Tensor;
const operators = @import("operators.zig");

pub fn exit() void {
    std.os.exit(1);
}

pub const Config = packed struct {
    dim: u32 = 1,
    hidden_dim: u32 = 1, // for ffn layers
    n_layers: u32 = 1, // number of layers
    n_heads: u32 = 1, // number of query heads
    n_kv_heads: u32 = 1, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: u32 = 1, // vocabulary size, usually 256 (byte-level)
    seq_len: u32 = 1, // max sequence length

    fn print(self: Config) void {
        std.debug.print("dim {any} .\n", .{self.dim});
    }
};

const Block = struct {
    rms_att_weight: Tensor(f32, 1), // dim
    rms_ffn_weight: Tensor(f32, 1), // dim

    // weights for matmuls. note dim == n_heads * head_size
    q_weight: Tensor(f32, 2), // dim, n_heads * head_size
    k_weight: Tensor(f32, 2), // dim, n_heads * head_size
    v_weight: Tensor(f32, 2), // dim, n_heads * head_size
    out_weight: Tensor(f32, 2), // dim, n_heads * head_size

    // ffn;
    weight1: Tensor(f32, 2), // hidden_dim, dim
    weight2: Tensor(f32, 2), // dim, hidden_dim
    weight3: Tensor(f32, 2), // hidden_dim, dim

    pub fn init(allocator: Allocator, config: Config) Block {
        const rms_att_weight = Tensor(f32, 1).init(allocator, .{config.dim}) catch exit();
        const rms_ffn_weight = Tensor(f32, 1).init(allocator, .{config.dim}) catch exit();

        const head_size = config.dim / config.n_heads;
        const qkv_shape = .{ config.dim, config.n_heads * head_size };
        const q_weight = Tensor(f32, 2).init(allocator, qkv_shape) catch exit();
        const k_weight = Tensor(f32, 2).init(allocator, qkv_shape) catch exit();
        const v_weight = Tensor(f32, 2).init(allocator, qkv_shape) catch exit();
        const out_weight = Tensor(f32, 2).init(allocator, qkv_shape) catch exit();

        const weight1 = Tensor(f32, 2).init(allocator, .{ config.hidden_dim, config.dim }) catch exit();
        const weight2 = Tensor(f32, 2).init(allocator, .{ config.dim, config.hidden_dim }) catch exit();
        const weight3 = Tensor(f32, 2).init(allocator, .{ config.hidden_dim, config.dim }) catch exit();

        return Block{
            .rms_att_weight = rms_att_weight,
            .rms_ffn_weight = rms_ffn_weight,
            .q_weight = q_weight,
            .k_weight = k_weight,
            .v_weight = v_weight,
            .out_weight = out_weight,
            .weight1 = weight1,
            .weight2 = weight2,
            .weight3 = weight3,
        };
    }

    pub fn deinit(self: Block) void {
        self.rms_att_weight.deinit();
        self.rms_ffn_weight.deinit();
        self.q_weight.deinit();
        self.k_weight.deinit();
        self.v_weight.deinit();
        self.out_weight.deinit();
        self.weight1.deinit();
        self.weight2.deinit();
        self.weight3.deinit();
    }
};

const State = struct {
    allocator: Allocator,
    x: Tensor(f32, 1), // activation at current time stamp (dim,)
    xb: Tensor(f32, 1), // same, but inside a residual branch (dim,)
    xb2: Tensor(f32, 1), // an additional buffer just for convenience (dim,)
    hb: Tensor(f32, 1), // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Tensor(f32, 1), // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Tensor(f32, 1), // query (dim,)
    k: Tensor(f32, 1), // key (dim,)
    v: Tensor(f32, 1), // value (dim,)
    att: Tensor(f32, 1), // buffer for scores/attention values (n_heads, seq_len)
    logits: Tensor(f32, 1), // output logits
    // kv cache
    key_cache: ArrayList(Tensor(f32, 2)),
    value_cache: ArrayList(Tensor(f32, 2)),
    // float* key_cache;   // (layer, seq_len, dim) #huge chunk of memory;
    // float* value_cache; // (layer, seq_len, dim)

    pub fn init(allocator: Allocator, _: Config) State {
        // TODO create?
        return State{
            .allocator = allocator,
            // .x = Tensor(f32, 1).init(allocator, .{config.dim}) catch unreachable,
        };
    }

    pub fn deinit(self: State) void {
        self.allocator.destroy(self.x);
    }
};

pub const Transformer = struct {
    config: Config,
    state: *State = undefined,
    token_embedding_table: ArrayList(Tensor(f32, 1)), // vocab_size * (dim)
    blocks: ArrayList(Block), // N number of blocks repeated
    rms_final_weight: Tensor(f32, 1), //(dim,)

    pub fn init(allocator: Allocator, config: Config) Transformer {
        const rms_final_weight = Tensor(f32, 1).init(allocator, .{config.dim}) catch unreachable;

        return Transformer{
            .config = config,
            .token_embedding_table = ArrayList(Tensor(f32, 1)).init(allocator),
            .blocks = ArrayList(Block).init(allocator),
            .rms_final_weight = rms_final_weight,
        };
    }

    pub fn deinit(self: *Transformer) void {
        for (self.token_embedding_table) |embediing| {
            embediing.deinit();
        }
        self.token_embedding_table.deinit();

        for (self.blocks) |block| {
            block.deinit();
        }
        self.blocks.deinit();
        self.rms_final_weight.deinit();
    }

    pub fn forward(self: *Transformer, token: i32, position: i32) void {
        const embedding = self.token_embedding_table.get(token);
        const state: *State = self.state;
        state.x.copyFrom(embedding);

        for (self.blocks, 0..) |block, index| {
            std.deubg.print("index: {any} {any}", .{ index, position });
            operators.rms_norm(state.xb, block.rms_att_weight);

            // qkv dot product;
            operators.matmul(state.q, state.xb, block.q_weight);
            operators.matmul(state.k, state.xb, block.k_weight);
            operators.matmul(state.v, state.xb, block.v_weight);

            // RoPE;

            operators.embed_rope();

            // multihead attention;

        }
    }
};

pub fn loadModel(allocator: Allocator, path: []const u8) !Transformer {
    // const config: Config = .{};

    var file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();

    const post = file.getPos();
    std.debug.print("post: {any}", .{post});
    const reader = file.reader();
    const config = try reader.readStruct(Config);

    // config.print();
    std.debug.print("config: {any}", .{config});

    const transformer = Transformer.init(allocator, config);

    return transformer;
}

test "init" {
    // const config: Config
    const b = Block.init(std.testing.allocator, .{});
    defer b.deinit();
    std.debug.print("b: {any}", .{b});

    var path = "model.bin";
    const transformer = loadModel(std.testing.allocator, path);
    std.debug.print("transformer: {any}", .{transformer});
}
