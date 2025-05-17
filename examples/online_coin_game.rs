use iterative_methods::online_game::play_dumb_coin_game;

fn main() {
    for _i in 0..10 {
        play_dumb_coin_game(20_000_000);
    }
}
