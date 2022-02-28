use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use streaming_iterator::*;

pub trait OnlineGame<C, A, L, F> {
    fn advance(&mut self);
    fn context(&self) -> C;
    fn respond(&self, a: &A) -> (L, F);
}

pub trait Player<C, F, A> {
    fn choose(&mut self, c: &C) -> A;
    fn update(&self, a: &A, f: &F);
}

pub struct OnlineGameTrace<G, P, C, A, L, F>
where
    G: OnlineGame<C, A, L, F>,
    P: Player<C, F, A>,
{
    game: G,
    player: P,
    last_record: Option<GameRecord<C, A, L, F>>,
}

pub struct GameRecord<C, A, L, F> {
    pub context: C,
    pub action: A,
    pub loss: L,
    pub feedback: F,
}

impl<G, P, C, A, L, F> OnlineGameTrace<G, P, C, A, L, F>
where
    G: OnlineGame<C, A, L, F>,
    P: Player<C, F, A>,
{
    pub fn new(game: G, player: P) -> OnlineGameTrace<G, P, C, A, L, F> {
        OnlineGameTrace {
            game,
            player,
            last_record: None,
        }
    }
}

impl<G, P, C, A, L, F> StreamingIterator for OnlineGameTrace<G, P, C, A, L, F>
where
    G: OnlineGame<C, A, L, F>,
    P: Player<C, F, A>,
{
    type Item = OnlineGameTrace<G, P, C, A, L, F>;
    fn advance(&mut self) {
        self.game.advance();
        let context = self.game.context();
        let action = self.player.choose(&context);
        let (loss, feedback) = self.game.respond(&action);
        self.player.update(&action, &feedback);
        self.last_record = Some(GameRecord {
            context,
            action,
            feedback,
            loss,
        });
    }
    fn get(&self) -> Option<&<Self as streaming_iterator::StreamingIterator>::Item> {
        Some(self)
    }
}
/*
pub struct RockPaperScissorsGame {
    pub house_play: RockPaperScissors,
}

pub enum RockPaperScissors {
    Rock,
    Paper,
    Scissors,
}

impl OnlineGame<(),> for RockPaperScissorsGame {
    fn advance(&mut self){

    }
    fn context(&self){

    }
}
*/
#[derive(Clone, Copy, PartialEq)]
pub enum Coin {
    Tails,
    Heads,
}

impl Coin {
    pub fn uniform<R>(rng: &mut R) -> Coin
    where
        R: Rng,
    {
        Coin::heads_probability(0.5, rng)
    }
    pub fn heads_probability<R>(p: f64, rng: &mut R) -> Coin
    where
        R: Rng,
    {
        if rng.gen::<f64>() < p {
            Coin::Heads
        } else {
            Coin::Tails
        }
    }
}

pub trait CoinGame {
    fn advance(&mut self);
    fn last_coin(&self) -> Coin;
}

impl<G> OnlineGame<(), Coin, f64, Coin> for G
where
    G: CoinGame,
{
    fn advance(&mut self) {
        self.advance()
    }
    fn context(&self) {}
    fn respond(&self, a: &Coin) -> (f64, Coin) {
        let loss = if self.last_coin() != *a { 1. } else { 0. };
        (loss, self.last_coin())
    }
}

pub struct FixedCoin {
    probability: f64,
    last_coin: Coin,
    rng: Pcg64,
}

impl FixedCoin {
    pub fn new(probability: f64) -> FixedCoin {
        let mut rng = Pcg64::from_entropy();
        let coin = Coin::heads_probability(probability, &mut rng);
        FixedCoin {
            probability,
            rng,
            last_coin: coin,
        }
    }
}

impl CoinGame for FixedCoin {
    fn advance(&mut self) {
        let coin = Coin::heads_probability(self.probability, &mut self.rng);
        self.last_coin = coin;
    }
    fn last_coin(&self) -> Coin {
        self.last_coin
    }
}

pub struct DumbCoinPlayer {}

impl Player<(), Coin, Coin> for DumbCoinPlayer {
    fn choose(&mut self, _c: &()) -> Coin {
        Coin::Heads
    }
    fn update(&self, _a: &Coin, _f: &Coin) {}
}

pub fn bla() -> Box<dyn CoinGame> {
    Box::new(FixedCoin::new(0.5))
}
pub fn bla2() -> Box<dyn OnlineGame<(), Coin, f64, Coin>> {
    Box::new(FixedCoin::new(0.5))
}

pub fn dumb_coin_game() -> OnlineGameTrace<FixedCoin, DumbCoinPlayer, (), Coin, f64, Coin> {
    let game = FixedCoin::new(0.5);
    let player = DumbCoinPlayer {};
    OnlineGameTrace::new(game, player)
}
