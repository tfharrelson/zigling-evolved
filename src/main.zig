const std = @import("std");
const zigling_evolved = @import("zigling_evolved");
const api = @import("proto/SC2APIProtocol.pb.zig");
const protobuf = @import("protobuf");
const ws = @import("websocket");

const sc2_host = "0.0.0.0";
const sc2_port = 5000;
pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try zigling_evolved.advancedPrint();

    const allocator = std.heap.page_allocator;

    const cmd = try std.process.getEnvVarOwned(allocator, "SC2_EXE");
    // TODO: parametrize port and url
    var buf: [4]u8 = undefined;
    const port_string = try std.fmt.bufPrint(&buf, "{}", .{sc2_port});
    const argv = [_][]const u8{ cmd, "-listen", sc2_host, "-port", port_string };
    var c = std.process.Child.init(&argv, allocator);
    try c.spawn();
    errdefer unsafe_kill(&c);

    // start a game using the protobuf api
    var sock = try connect_to_game(allocator, 0);
    defer sock.deinit();

    // TODO: change this to an array list once i do the painstaking chore
    // of moving all managed lists in zig-protobuf to unmanaged array lists
    var players = std.array_list.Managed(api.PlayerSetup).init(allocator);
    defer players.deinit();

    try players.append(api.PlayerSetup{
        .type = api.PlayerType.Participant,
        .race = api.Race.Zerg,
    });
    try players.append(api.PlayerSetup{
        .type = api.PlayerType.Computer,
    });

    const map_dir = try std.process.getEnvVarOwned(allocator, "SC2_MAP_DIR");
    const map_path: []const u8 = try std.fmt.allocPrint(allocator, "{s}/AutomatonLE.SC2Map", .{map_dir});
    const map = api.RequestCreateGame.Map_union{ .local_map = .{ .map_path = protobuf.ManagedString{ .Const = map_path } } };
    const start_game_request: api.Request = .{ .request = .{ .create_game = .{ .player_setup = players, .Map = map } } };
    defer start_game_request.deinit();

    const start_game_response = try send(&sock, start_game_request);
    std.debug.print("wrote to server\n", .{});

    if (start_game_response.response) |vresp| {
        switch (vresp) {
            .create_game => std.debug.print("found create game response! error details: {any}\n", .{vresp.create_game.error_details}),
            else => std.debug.print("found something weird\n", .{}),
        }
    }

    // TODO: add other processes here like the 2 bots
    const default_client_ports = std.array_list.Managed(api.PortSet).init(allocator);
    const join_game_request: api.Request = .{ .request = .{ .join_game = .{ .participation = .{ .race = api.Race.Zerg }, .client_ports = default_client_ports, .options = .{ .raw = true } } } };
    std.debug.print("attempting to join game with server\n", .{});
    const join_game_response = try send(&sock, join_game_request);

    if (join_game_response.response) |vresp| {
        switch (vresp) {
            .join_game => {
                if (vresp.join_game.error_details) |deets| {
                    std.debug.print("found create game response! error: {?d}\n", .{vresp.join_game.@"error"});
                    std.debug.print("found create game response! error details: {s}\n", .{deets.getSlice()});
                } else {
                    std.debug.print("found create game response! error details: None\n", .{});
                }
            },
            else => std.debug.print("found something weird", .{}),
        }
    }

    // start game loop?
    var obs = try observe(&sock);
    while (!is_game_over(obs)) {
        const step_request: api.Request = .{ .request = .{ .step = .{} } };
        const step_response = try send(&sock, step_request);
        std.debug.print("loop number = {?d}\n", .{step_response.response.?.step.simulation_loop});
        obs = try observe(&sock);
    }
    std.debug.print("game over!\n", .{});

    _ = try c.kill();
    _ = try c.wait();
}

fn unsafe_kill(child: *std.process.Child) void {
    _ = child.kill() catch unreachable;
}

fn observe(sock: *ws.Client) !api.ResponseObservation {
    const info_request: api.Request = .{ .request = .{ .observation = .{ .disable_fog = false } } };
    const raw_info_response = try send(sock, info_request);
    const info_response = raw_info_response.response orelse unreachable;
    return info_response.observation;
}
fn send(sock: *ws.Client, req: api.Request) !api.Response {
    const alloc = std.heap.page_allocator;
    try sock.write(try req.encode(alloc));
    const msg = try sock.read();
    const resp = try api.Response.decode(msg.?.data, alloc);
    std.debug.print("response status = {?d}\n", .{resp.status});
    return resp;
}

fn is_game_over(obs: api.ResponseObservation) bool {
    return obs.player_result.items.len > 0;
}

fn connect_to_game(allocator: std.mem.Allocator, attempt_num: u8) !ws.Client {
    var client = ws.Client.init(allocator, .{
        .host = sc2_host,
        .port = sc2_port,
    }) catch |err| {
        if (attempt_num < 5) {
            std.Thread.sleep(2000000000);
            return connect_to_game(allocator, attempt_num + 1);
        } else {
            return err;
        }
    };
    std.debug.print("Connected!\n", .{});

    client.handshake("/sc2api", .{ .timeout_ms = 1000 }) catch |err| {
        if (attempt_num < 5) {
            std.Thread.sleep(2000000000);
            return connect_to_game(allocator, attempt_num + 1);
        } else {
            return err;
        }
    };
    errdefer client.deinit();
    std.debug.print("handshake succeeded.\n", .{});
    return client;
}

test "simple test" {
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(std.testing.allocator); // Try commenting this out and see if zig detects the memory leak!
    try list.append(std.testing.allocator, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
