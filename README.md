# Blondie24: Evolving Neural Networks to Play Checkers in C++

This is a C++ implementation of the 1999 AI paper, *"Evolving Neural Networks to Play Checkers Without Relying on Expert Knowledge"* by Kumar Chellapilla and David B. Fogel. 

The goal of this project is to create a neural network with neuroevolution from scratch  in C++ that learns to play checkers without any expert human knowledge.

## Key Features
* **Zero Expert Knowledge:** The only inputs given to the neural network are the current pieces on the board and the piece differential(sum of all the piece values on the bode)
* **Evolutionary Learning:** A population of neural networks competes in a tournament. The best performers survive and create offspring through random mutation of their weights and biases.
* **Built from Scratch:** The neural network, matrix math (via Eigen), and the game tree search algorithms are implemented directly in C++.

---

## 1. Board Representation

To make the AI fast and efficient, the 8x8 checkers board is squashed down into a 1D array of 32 squares. Checkers only uses the dark squares, so we completely ignore the light squares to save memory and processing time.

### The 32-Square Indexing

In our engine, the board is numbered 1 through 32 (mapped to indices 0 to 31 in the C++ array). Player 1 (the network) starts at the bottom, and Player 2 (the opponent) starts at the top.

```text
      Red Side (Player 1)
|---|---|---|---|---|---|---|---|
|   | 0 |   | 1 |   | 2 |   | 3 | <-------- 0
|---|---|---|---|---|---|---|---|
| 4 |   | 5 |   | 6 |   | 7 |   | <-------- 1
|---|---|---|---|---|---|---|---|
|   | 8 |   | 9 |   | 10|   | 11| <-------- 2
|---|---|---|---|---|---|---|---|
| 12|   | 13|   | 14|   | 15|   | <-------- 3
|---|---|---|---|---|---|---|---|
|   | 16|   | 17|   | 18|   | 19| <-------- 4
|---|---|---|---|---|---|---|---|
| 20|   | 21|   | 22|   | 23|   | <-------- 5
|---|---|---|---|---|---|---|---|
|   | 24|   | 25|   | 26|   | 27| <-------- 6
|---|---|---|---|---|---|---|---|
| 28|   | 29|   | 30|   | 31|   | <-------- 7

```

### 1. The Starting Board (8x8)

Here is how the starting pieces look on a traditional 8x8 board. The unplayable light squares are marked with a `.`. 

* **-1** : Player 2 (Opponent)
* **0** : Empty Dark Square
* **1** : Player 1 (Network)

```text
      Opponent Side (Player 2)
|---|---|---|---|---|---|---|---|
| . |-1 | . |-1 | . |-1 | . |-1 |
|---|---|---|---|---|---|---|---|
|-1 | . |-1 | . |-1 | . |-1 | . |
|---|---|---|---|---|---|---|---|
| . |-1 | . |-1 | . |-1 | . |-1 |
|---|---|---|---|---|---|---|---|
| 0 | . | 0 | . | 0 | . | 0 | . |
|---|---|---|---|---|---|---|---|
| . | 0 | . | 0 | . | 0 | . | 0 |
|---|---|---|---|---|---|---|---|
| 1 | . | 1 | . | 1 | . | 1 | . |
|---|---|---|---|---|---|---|---|
| . | 1 | . | 1 | . | 1 | . | 1 |
|---|---|---|---|---|---|---|---|
| 1 | . | 1 | . | 1 | . | 1 | . |
|---|---|---|---|---|---|---|---|
       Network Side (Player 1)

Indices 00-11 (Opponent): [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
Indices 12-19 (Empty)   : [ 0,  0,  0,  0,  0,  0,  0,  0]
Indices 20-31 (Network) : [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

Full Array: 
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

```

## 2. Move Generation

Generating legal moves on a 1D array requires a different approach than a standard 2D grid. The move generation pipeline is broken down into three main components:

### Navigating the 1D Board (`get_step`)
Because the 32 playable squares alternate positions row by row, the mathematical offset for a diagonal move changes depending on whether the piece is currently on an "even" row (e.g., indices 0-3) or an "odd" row (e.g., indices 4-7). The `get_step` function takes a starting index and a direction (Up-Left, Up-Right, Down-Left, Down-Right) and calculates the exact array index of the destination square, returning `-1` if the move slides off the board.

### Jumps, Slides, and Chains (`get_legal_moves` & `find_jump_chains`)
The engine scans the board for pieces belonging to the current player and calculates their valid diagonal moves:
* **Slides:** Moving to an adjacent empty square.
* **Jumps & Chains:** If an adjacent square contains an opponent and the square directly behind it is empty, it is a valid jump. 
* **Recursion:** If a jump would in turn place the jumping checker in position for another jump, that jump must also be played, and so forth, until no further jumps are available for that piece. The `find_jump_chains` function uses recursion to temporarily update the board state and trace every possible multi-jump path to its end.

### The Mandatory Jump Rule
The engine strictly enforces standard checkers rules: Whenever a jump is available, it must be played in preference to a move that does not jump. However, when multiple jump moves are available, the player has the choice of which jump to conduct. If `get_legal_moves` detects any available jumps, it discards all standard slides and forces the network to pick from the list of jumps.

---

## 3. The Neural Network

The network is a feedforward neural network with **two hidden layers**, taking in the board state and outputting a single score between -1 and 1. A positive score means the position favors the network (Player 1); a negative score favors the opponent.

### Architecture
```
Input (33)  →  Hidden Layer 1 (40)  →  Hidden Layer 2 (10)  →  Output (1)
```

The input vector is the 32 board squares plus a **bias node**. Each hidden layer also gets an appended bias node before passing into the next layer, so the actual weight matrices are shaped `33×40`, `41×10`, and `11×1`. All hidden activations use `tanh`.

The **piece differential** (sum of all values on the board) bypasses the layers entirely and is added directly to the final pre-activation value before the output `tanh`. This is an explicit shortcut from the paper that gives the network a hard-coded sense of material advantage.

### King Representation

Rather than a fixed king value, each network evolves its own **K parameter**, a floating-point value that represents how much a king is worth relative to a man. Kings are stored in the board array as `±K` (where the sign indicates player), so the network's evaluation of a king's worth is literally baked into the board encoding it reads.

---

## 4. Neuroevolution

The networks are trained entirely through **evolution** — no gradients, no backpropagation.

### Self-Adaptive Mutation (EP-style)

Each network carries its own **per-weight mutation rates** (`sigma` matrices, one per weight matrix). When a network reproduces, it doesn't just mutate its weights — it also mutates its own sigmas first. This is the EP (Evolutionary Programming) strategy from the paper:
```
σ'ᵢ  = σᵢ · exp(τ · N(0,1))       # mutate the mutation rate
w'ᵢ  = wᵢ + σ'ᵢ · N(0,1)          # mutate the weight using the new rate
```

where `τ = 1 / sqrt(2 * sqrt(n))` and `n = 1741` (total number of weights). All sigmas are initialized to `0.05` per the paper. `K` is also mutated each generation and clamped to the range `[1.0, 3.0]`.

### The Evolutionary Loop

The training loop runs for **250 generations** with a population of **15 networks**:

1. Each of the 15 parents produces one child via `replicate()`, giving a combined pool of 30.
2. Every network in the pool plays **5 games** against randomly selected opponents from the same pool.
3. Scores are tallied: `+1` for a win, `-2` for a loss, `0` for a draw. The asymmetric scoring penalizes losses heavily, rewarding networks that avoid getting beaten.
4. The top 15 networks by total score survive to become the next generation's parents.

The game evaluation loop is parallelized with **OpenMP** (`#pragma omp parallel for`), with `thread_local` RNG to prevent data races across threads.

---

## 5. Game Tree Search

Each network picks its moves using **minimax with alpha-beta pruning**, searching 4 plies deep (the root loop counts as 1 ply, passing `depth=3` to the recursive function).

### Alpha-Beta Pruning

The standard alpha-beta algorithm is implemented: the maximizer (Player 1) tracks `alpha`, the minimizer (Player 2) tracks `beta`, and any branch where `beta <= alpha` is pruned. At the root, alpha and beta are tightened across moves to allow better pruning deeper in the tree.

**Move Ordering:** Moves are sorted before evaluation to improve pruning efficiency, the network's moves are sorted by destination index ascending (toward the opponent's king row), and the opponent's moves are sorted descending.

### Forced Move & Quiescence Extensions

To avoid evaluating positions mid-sequence or at artificially quiet horizons, two search extensions are applied when `depth == 0`:

* **Forced move extension:** If a forced count (number of single-legal-move plies encountered) is odd, 1 extra ply is added to reach an even position boundary.
* **Active state extension:** If the leaf node has an available jump, 2 extra plies are added so the search resolves the capture sequence before evaluating.

Extensions only trigger once per search path (`in_extension` flag), preventing exponential blowup.

---

## 6. Building and Running

### Dependencies

* **Eigen 3** — for matrix math. Install via your package manager or download from [eigen.tuxfamily.org](https://eigen.tuxfamily.org).
* A C++17-compatible compiler (GCC or Clang).
* **OpenMP** (optional but recommended for parallel training).

### Compile
```bash
# With OpenMP (recommended)
g++ -O2 -std=c++17 -fopenmp -I /path/to/eigen blondie24.cpp -o blondie24

# Without OpenMP
g++ -O2 -std=c++17 -I /path/to/eigen blondie24.cpp -o blondie24
```

### Run
```bash
./blondie24
```

Training will print progress each generation. After 250 generations, the best network's weights and sigmas are saved to two output files.

---

## 7. Output Files

After training completes, two files are written to the working directory:

* **`best_network.bin`** — A compact binary file containing `K`, the three weight matrices (`w1`, `w2`, `w3`), and the three sigma matrices (`sigma1`, `sigma2`, `sigma3`). Each matrix is preceded by its row/column dimensions as `int`s, followed by raw `double` values in column-major order (Eigen default).

* **`best_network.txt`** — A human-readable version of the same data, with matrix dimensions and values printed at 8 decimal places of precision. Useful for debugging or porting weights to another language.
