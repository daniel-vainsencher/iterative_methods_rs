# Design for demos based on hedge + reduction

## Design Hedge player

## How to create demos needed for post
### Spec of plots
#### Format
- Timeline plots:
    - horizonal axis is single pixel per step, we choose to run 500
      rounds so we can show the whole experiment (hope that learning
      happens fast enough!)
    - Subpane 1:
        - coin result: 0 for Tails, 1 for head.
        - system probability of heads outcome
        - player probability of heads action
        - players' accrued cost
    - Subpane 2:
        - best constant action average cost so far
        - player average cost so far
        - difference between the above (different scale?)
- Expected cost plot? not clear.
    - Probability of penalty is `a_t * (1-p_t) + (1-a_t) * (p_t)`
    - Thus a bilinear, constant over sets `a_t=0.5` and at `p_t=0.5`.
#### Different variants
Timeline plot variants:
- fixed constant action under a constant assumption
  situation.
- hedge with constant p_t
- hedge with p_t that flips every 100 rounds
- hedge with p_t that has negative autocorrelation
- hedge with p_t that changes smoothly from 0 to 1

### Design of needed games
- define trait CoinGame
- impl OnlineGame for CoinGame
    - C = ()
    - F = {Tails, Heads}
    - L = {0,1}
    - advance uses internal state (remains abstract), computes next p_t and F.
    - response decides loss=1 if action != F
- Structs with CoinGame impls:
    - p_t is just an instance variable.
    - p_t = 0.75 if previous F was Tails, 0.25 otherwise (negative correlation) 
    - p_t = 1/t

### What data to output?
For every game round, pick up the 
- action, 
- feedback (so we can compute losses of arbitray (best in hindsight)
actions), 
- loss accrued


### How to get plots?
- [x] impl StreamIterator for CoinGame over steps of game.
- [x] define trait OnlineGameTrace with functions to get the context,
      feedback and loss.
- [x] Compute average loss. Implement IterativeMethod? solution would
      be player's probability.
- impl<T> YamlDataType for OnlineGameTrace as creating a dictionary,
  as long as it is defined for C, F, L.
- wrap `write_yaml_documents(v_iter, String::from(test_file_path))`
  around the stream.
- Read them from python
- Create plot using plotly, similarly to the existing demos
- Alternative: 
    - accumulate iterator of Struct into Struct of Vectors<Option<T>> (maybe exists in some SOA crate?),
    - call python functions to convert into:
        - python dictionary of numpy vectors
        - pandas dataframe 
        - plot.
