#include <Eigen/Dense> // for matrix math
#include <algorithm>
#include <array> 
#include <cmath> // tanh, exp, sqrt
#include <iostream>
#include <fstream> 
#include <random>   // random number generation 
#include <vector>   

thread_local std::mt19937 rng(std::random_device{}());//for openmp parallism later

const std::array<double, 32> initial_board = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
    0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1, 1, 1};
const int UL = 0, UR = 1, DL = 2, DR = 3;//represent directions as int to be faster
const float KING_VALUE = 1.5f; // fixed king value 

int get_step(int i, int direction) {
  bool is_even = ((i >> 2) & 1) == 0;//check if row is an even row(same as dividing by 4 and checking mod 2, switched to bitwise to be faster)

  if (direction == 0) { // up left
    if (is_even) {
      return i - 4;//moving up left in an even row is subtracting 4
    } else {
      if (i % 4 == 0) { //in odd rows we have left edges so we cant always make left moves here
        return -1;
      } else {
        return i - 5;//when valid up left move exist in an odd row, its subtracting 5
      }
    }
  }

  else if (direction == 1) { // up right
    if (is_even) {
      if ((i + 1) % 4 == 0) {//even rows have right edges so we cant make right moves in a couple of cases
        return -1;
      }

      else {//when we can moving up right in even is subtracting 3
        return i - 3;
      }
    } else {
      return i - 4;//for odd rows we can always move up right by subtracting 4
    }
  }
  
  
  else if (direction == 2) { // down left
    if (is_even) {
      return i + 4;
    } else {
      if (i % 4 == 0) { 
        return -1;
      } else {
        return i + 3;
      }
    }
  } else if (direction == 3) { // down right
    if (is_even) {
      if ((i + 1) % 4 == 0) {
        return -1;
      } else {
        return i + 5;
      }
    } else {
      return i + 4;
    }
  }

  return -1;
}

struct GameResult {
  bool is_over;
  int status; // 0 = ongoing, 1= win,-1 = loss, 2=tie
};

struct Move{
  int start;
  int end;
  std::vector<int> captured; // all captured piece indices (empty if slide)
  bool is_jump; // true if jump, false if slide
};



GameResult check_game_over(const std::array<double, 32> &board_state, int moves_played) {
  // first check if 200 move limit reached(100 per player)
  if (moves_played >= 200) {
    return {true, 2};
  }

  // if network has no pieces left, network loses
  bool network_pieces = false;
  bool opponent_pieces = false;
  for (int idx = 0; idx < 32; idx++) {
    if (board_state[idx] >= 1) {
      network_pieces = true;
      break;}
      }

  for (int idx = 0; idx < 32; idx++) {
    if (board_state[idx] <= -1) {
      opponent_pieces = true;
    break;}
      }

  if (! network_pieces){
    return {true, -1};}
  else if(! opponent_pieces){
    return {true, 1};}

  return {false, 0};
}

class NeuralNetwork{
  public:
    Eigen::MatrixXd w1, w2, w3;
    //per-weight mutation rates: each weight has its own sigma
    //same shape as the weight matrices they mutate
    Eigen::MatrixXd sigma1, sigma2, sigma3;

    //mutation constant from the paper (n = 1741 total weights)
    //tau = 1/sqrt(2*sqrt(n)): sqrt(1741)=41.72, 2*41.72=83.45, sqrt(83.45)=9.135
    static constexpr double TAU = 1.0 / 9.135;  // 1/sqrt(2*sqrt(1741)) ≈ 0.1095

    //default:random weights
    NeuralNetwork() {
      w1 = Eigen::MatrixXd::Random(33, 40) * 0.2;
      w2 = Eigen::MatrixXd::Random(41, 10) * 0.2;
      w3 = Eigen::MatrixXd::Random(11, 1) * 0.2;
      //all sigmas start at 0.05 per the paper
      sigma1 = Eigen::MatrixXd::Constant(33, 40, 0.05);
      sigma2 = Eigen::MatrixXd::Constant(41, 10, 0.05);
      sigma3 = Eigen::MatrixXd::Constant(11, 1, 0.05);
    }

    //with specific values(when using replicate)
    NeuralNetwork(Eigen::MatrixXd w1, Eigen::MatrixXd w2, Eigen::MatrixXd w3,
                  Eigen::MatrixXd sigma1, Eigen::MatrixXd sigma2, Eigen::MatrixXd sigma3)
      : w1(w1), w2(w2), w3(w3), sigma1(sigma1), sigma2(sigma2), sigma3(sigma3) {}

    double predict(const std::array<double, 32> &board_state) {
      //piece difference = sum of all 32 squares
      double piece_diff = 0.0;
      for (int i = 0; i < 32; i++){
        piece_diff += board_state[i];}

      //build input vector
      Eigen::VectorXd input(33);
      for (int i = 0; i < 32; i++){
        input(i) = board_state[i];}
      input(32) = 1.0;

      //layer 1: input(33) * w1 -> 40
      Eigen::VectorXd a1 = (input.transpose() * w1).transpose();
      a1 = a1.array().tanh();

      //layer2: input(41)*w2(41*10) -> 10
      Eigen::VectorXd a1_biased(41);
      a1_biased.head(40) = a1;
      a1_biased(40) = 1.0;
      Eigen::VectorXd a2 = (a1_biased.transpose() * w2).transpose();
      a2 = a2.array().tanh();

      // layer3: a2(10) + bias -> 11 inputs*w3(11x1) + piece_diff -> output
      // piece_diff connects directly to output with implicit weight of 1
      Eigen::VectorXd a2_full(11);
      a2_full.head(10) = a2;
      a2_full(10) = 1.0;
      double output = std::tanh((a2_full.transpose() * w3)(0, 0) + piece_diff);

      return output;
    }


    NeuralNetwork replicate(){
      std::normal_distribution<double> normal(0.0, 1.0);
      //each weight's sigma gets its own random normal, scaled by tau
      auto mutate_sigma = [&](const Eigen::MatrixXd &s) {
        return s.unaryExpr([&](double si) {
          return si * std::exp(TAU * normal(rng));
        });
      };

      //mutate all three sigma matrices
      Eigen::MatrixXd new_sigma1 = mutate_sigma(sigma1);
      Eigen::MatrixXd new_sigma2 = mutate_sigma(sigma2);
      Eigen::MatrixXd new_sigma3 = mutate_sigma(sigma3);

      //mutate weights: w'_i = w_i + sigma'_i * N_i(0,1)
      //each weight gets its own random perturbation scaled by its own sigma
      Eigen::MatrixXd new_w1 = w1 + new_sigma1.unaryExpr([&](double si) { return si * normal(rng); });
      Eigen::MatrixXd new_w2 = w2 + new_sigma2.unaryExpr([&](double si) { return si * normal(rng); });
      Eigen::MatrixXd new_w3 = w3 + new_sigma3.unaryExpr([&](double si) { return si * normal(rng); });

      return NeuralNetwork(new_w1, new_w2, new_w3, new_sigma1, new_sigma2, new_sigma3);
    }
};


// multijump struct replaces simple move for jump chains
// a jump chain stores every captured index along the path
struct JumpChain {
  int start;
  int end;
  std::vector<int> captured; // all pieces removed in this chain
};

//finds all jump chains from a position
void find_jump_chains(const std::array<double, 32> &board, int pos, double piece,
                      int current_player,
                      std::vector<int> &captured_so_far,
                      std::vector<JumpChain> &results, int original_start) {

  //figure out which directions this piece can move
  std::vector<int> directions;
  if (piece == 1 || std::abs(piece) == KING_VALUE) {
    directions.push_back(UL);
    directions.push_back(UR);
  }
  if (piece == -1 || std::abs(piece) == KING_VALUE) {
    directions.push_back(DL);
    directions.push_back(DR);
  }

  bool found_more = false;

  for (int dir : directions) {
    int mid = get_step(pos, dir);
    if (mid == -1 || mid < 0 || mid > 31) continue;//filter out invalid steps

    // piece on mid must be an opponent piece
    bool is_opponent;
    if (current_player == 1)
      is_opponent = (board[mid] == -1 || board[mid] == -KING_VALUE);
    else
      is_opponent = (board[mid] == 1 || board[mid] == KING_VALUE);

    if (!is_opponent) continue;//skip if its not an opponent piece

    // check if we already captured this piece in this chain, we dont want to capture a piece twice
    bool already_captured = false;
    for (int c : captured_so_far) {
      if (c == mid) { already_captured = true; break; }
    }
    if (already_captured) continue;

    // land must be empty or be our starting square
    int land = get_step(mid, dir);
    if (land == -1 || land < 0 || land > 31) continue;
    if (board[land] != 0 && land != pos) continue;

    //king promotion mid chain
    bool promotes = (piece == 1 && land <= 3) || (piece == -1 && land >= 28);
    if (promotes) {
      captured_so_far.push_back(mid);
      results.push_back({original_start, land, captured_so_far});
      captured_so_far.pop_back();
      found_more = true;
      continue;
    }

    // valid jump found, recurse
    found_more = true;
    captured_so_far.push_back(mid);

    // make temporary board for recursion (remove captured piece)
    std::array<double, 32> temp_board = board;
    temp_board[mid] = 0;
    temp_board[pos] = 0;
    temp_board[land] = piece;

    find_jump_chains(temp_board, land, piece, current_player,
                     captured_so_far, results, original_start);

    captured_so_far.pop_back();
  }

  //if no more jumps found and we've captured at least one piece, save this chain
  if (!found_more && !captured_so_far.empty()) {
    JumpChain chain;
    chain.start = original_start;
    chain.end = pos;
    chain.captured = captured_so_far;
    results.push_back(chain);
  }
}
//check and get all legal moves
std::vector<Move> get_legal_moves(const std::array<double, 32> &board_state,
                                   int current_player) {
  std::vector<Move> slides;
  std::vector<Move> jumps;

  for (int i = 0; i < 32; i++) {
    double piece = board_state[i];

    //check if this piece belongs to current player
    bool is_mine;
    if (current_player == 1)
      is_mine = (piece == 1 || piece == KING_VALUE);
    else
      is_mine = (piece == -1 || piece == -KING_VALUE);

    if (!is_mine) continue;

    //figure out which directions this piece can move
    std::vector<int> directions;
    if (piece == 1 || std::abs(piece) == KING_VALUE) {
      directions.push_back(UL);
      directions.push_back(UR);
    }
    if (piece == -1 || std::abs(piece) == KING_VALUE) {
      directions.push_back(DL);
      directions.push_back(DR);
    }

    for (int dir : directions) {
      int mid = get_step(i, dir);
      if (mid == -1 || mid < 0 || mid > 31) continue;//again skip invalid steps

      //if mid is empty, then we slide
      if (board_state[mid] == 0) {
        slides.push_back({i, mid, {}, false});
        continue;
      }

      //if there's an opponent in mid, then its a jump'
      bool is_opponent;
      if (current_player == 1)
        is_opponent = (board_state[mid] == -1 || board_state[mid] == -KING_VALUE);
      else
        is_opponent = (board_state[mid] == 1 || board_state[mid] == KING_VALUE);

      if (is_opponent) {
        int land = get_step(mid, dir);
        if (land != -1 && land >= 0 && land <= 31 && board_state[land] == 0) {
          //we've found a single jump, now we check for chains
          std::vector<int> captured_so_far;
          std::vector<JumpChain> chains;
          captured_so_far.push_back(mid);

          std::array<double, 32> temp_board = board_state;
          temp_board[mid] = 0;
          temp_board[i] = 0;
          temp_board[land] = piece;

          find_jump_chains(temp_board, land, piece, current_player,
                           captured_so_far, chains, i);

          if (chains.empty()) {
            // single jump only, no chain found
            jumps.push_back({i, land, {mid}, true});
          } else {
            //add all chains as moves as well as the capture lists
            for (auto &chain : chains) {
              jumps.push_back({chain.start, chain.end, chain.captured, true});
            }
          }
        }
      }
    }
  }

  //we must jump if there's a jump'
  if (!jumps.empty()) return jumps;
  return slides;
}


std::array<double, 32> simulate_move(std::array<double, 32> board,
                                      const Move &move) {
  // grab the piece that's moving
  double piece = board[move.start];

  // move the piece
  board[move.end] = piece;
  board[move.start] = 0;

  // remove all captured pieces (handles single jumps and chains)
  if (move.is_jump) {
    for (int idx : move.captured) {
      board[idx] = 0;
    }
  }

  //promote kings
  if (piece == 1 && move.end <= 3) {
    board[move.end] = KING_VALUE;
  }
  //promote kings
  else if (piece == -1 && move.end >= 28) {
    board[move.end] = -KING_VALUE;
  }

  return board;
}

//minimax with alpha beta pruning
double minimax(const std::array<double, 32> &board_state, int depth,
               bool maximizing_player, NeuralNetwork &network,
               int moves_played, int forced_count = 0,
               bool in_extension = false,
               double alpha = -1e9, double beta = 1e9) {

  // base case: check if game is over
  GameResult result = check_game_over(board_state, moves_played);
  if (result.is_over) {
    if (result.status == 1) return 1.0;       // win
    else if (result.status == -1) return -1.0; // loss
    else return 0.0;                            // tie
  }


  if (depth == 0) {
    if (!in_extension) {
      int extension = 0;

      // condition 1: odd forced count means we are 1 ply short of an even extension
      if (forced_count % 2 != 0) {
        extension += 1;
      }

      // condition 2: active state (forced jump at leaf)
      int cur_player = maximizing_player ? 1 : -1;
      auto leaf_moves = get_legal_moves(board_state, cur_player);
      if (!leaf_moves.empty() && leaf_moves[0].is_jump) {
        extension += 2;
      }

      if (extension > 0) {
        depth = extension;
        in_extension = true;
      } else {
        return network.predict(board_state);
      }
    } else {
      return network.predict(board_state);
    }
  }

  if (maximizing_player) {
    //network's turn (player 1), wants highest score
    std::vector<Move> legal_moves = get_legal_moves(board_state, 1);

    //sort moves(this is supposed to make alpha beta pruning more efficient)
    //because closer to king row moves are generally better
    //but i havent verified yet, included in my futute plans
    std::sort(legal_moves.begin(), legal_moves.end(),
              [](const Move &a, const Move &b) { return a.end < b.end; });

    // no moves = trapped = loss
    if (legal_moves.empty()) return -1.0;

  
    // forced move: only one legal move, dont decrement depth
    // this inherently gives +1 ply evaluating 1 move (super fast, no explosion)
    bool is_forced = (legal_moves.size() == 1);
    int next_depth = is_forced ? depth : depth - 1;
    int next_forced = is_forced ? forced_count + 1 : forced_count;

    double max_eval = -1e9;
    for (auto &move : legal_moves) {
      auto new_board = simulate_move(board_state, move);
      double eval = minimax(new_board, next_depth, false, network,
                            moves_played + 1, next_forced, in_extension, alpha, beta);
      max_eval = std::max(max_eval, eval);

      // alpha-beta: if we found something >= beta, minimizer above
      // would never let us reach here, prune the rest
      alpha = std::max(alpha, eval);
      if (beta <= alpha) break;
    }
    return max_eval;

  } else {
    // opponent's turn (player -1), wants lowest score
    std::vector<Move> legal_moves = get_legal_moves(board_state, -1);

    // sort moves
    std::sort(legal_moves.begin(), legal_moves.end(),
              [](const Move &a, const Move &b) { return a.end > b.end; });

    // no moves = opponent trapped = win for network
    if (legal_moves.empty()) return 1.0;

    // forced move: same logic, dont consume depth for non-decisions
    bool is_forced = (legal_moves.size() == 1);
    int next_depth = is_forced ? depth : depth - 1;
    int next_forced = is_forced ? forced_count + 1 : forced_count;

    double min_eval = 1e9;
    for (auto &move : legal_moves) {
      auto new_board = simulate_move(board_state, move);
      double eval = minimax(new_board, next_depth, true, network,
                            moves_played + 1, next_forced, in_extension, alpha, beta);
      min_eval = std::min(min_eval, eval);

      // beta tracks the best the minimizer can guarantee
      // if something <= alpha, maximizer above wouldnt pick this path
      beta = std::min(beta, eval);
      if (beta <= alpha) break;
    }
    return min_eval;
  }
}


// play a full game between two networks
// returns: +1 if network1 wins, -2 if network1 loses, 0 for tie
int play_game(NeuralNetwork &network1, NeuralNetwork &network2) {
  auto board = initial_board;
  int current_player = 1; // network1 goes first
  int moves_played = 0;

  while (true) {
    // check if game is over
    GameResult result = check_game_over(board, moves_played);
    if (result.is_over) {
      if (result.status == 1) return 1;       // network1 wins
      else if (result.status == -1) return -2; // network1 loses
      else return 0;                            // tie
    }

    // get the right network for current player
    NeuralNetwork &current_net = (current_player == 1) ? network1 : network2;

    // find legal moves
    std::vector<Move> legal_moves = get_legal_moves(board, current_player);

    // no moves = trapped = loss for current player
    if (legal_moves.empty()) {
      return (current_player == 1) ? -2 : 1;
    }

    // pick the best move using depth-4 minimax
    // root loop is 1 ply, so pass depth=3 for remaining 3 ply = 4 total
    Move best_move = legal_moves[0];
    if (current_player == 1) {
      // network1 maximizes, track alpha across root moves
      double best_score = -1e9;
      double root_alpha = -1e9;
      for (auto &move : legal_moves) {
        auto new_board = simulate_move(board, move);
        double score = minimax(new_board, 3, false, network1,
                               moves_played + 1, 0, false, root_alpha, 1e9);
        if (score > best_score) {
          best_score = score;
          best_move = move;
        }
        // tighten alpha at root so deeper searches can prune more
        root_alpha = std::max(root_alpha, score);
      }
    } else {
      // network2 minimizes — low score = bad for player 1 = good for player 2
      // predict() always evaluates from player 1's perspective
      double best_score = 1e9;
      double root_beta = 1e9;
      for (auto &move : legal_moves) {
        auto new_board = simulate_move(board, move);
        double score = minimax(new_board, 3, true, network2,
                               moves_played + 1, 0, false, -1e9, root_beta);
        if (score < best_score) {
          best_score = score;
          best_move = move;
        }
        // tighten beta at root
        root_beta = std::min(root_beta, score);
      }
    }

    // make the best move
    board = simulate_move(board, best_move);
    moves_played += 1;
    current_player *= -1; // swap players
  }
}


int main() {
  const int POP_SIZE = 15;
  const int NUM_GENERATIONS = 250;
  const int GAMES_PER_NETWORK = 5;

  //initialize population of 15 random networks
  std::vector<NeuralNetwork> population;
  for (int i = 0; i < POP_SIZE; i++) {
    population.push_back(NeuralNetwork());
  }

  // distribution for picking random opponents
  std::uniform_int_distribution<int> opponent_dist(0, 2 * POP_SIZE - 2);

  for (int gen = 0; gen < NUM_GENERATIONS; gen++) {
    // each parent produces one child
    std::vector<NeuralNetwork> children;
    for (auto &parent : population) {
      children.push_back(parent.replicate());
    }

    // combine parents + children = 30 networks
    std::vector<NeuralNetwork> combined;
    combined.reserve(2 * POP_SIZE);
    for (auto &net : population) combined.push_back(net);
    for (auto &net : children) combined.push_back(net);

    // evaluate: each network plays 5 games against random opponents
    std::vector<int> scores(combined.size(), 0);

    #pragma omp parallel for schedule(dynamic)//make the evaluation parallel

    for (int i = 0; i < (int)combined.size(); i++) {
      for (int g = 0; g < GAMES_PER_NETWORK; g++) {
        // pick a random opponent thats not ourselves
        // pick a random opponent thats not ourselves
        int opp = opponent_dist(rng);
        if (opp >= i) opp++; // skip self

        // Play the game and get the result for network 1
        int p1_score = play_game(combined[i], combined[opp]);
        
        // Calculate the opponent's score based on the result
        int p2_score = 0;
        if (p1_score == 1) p2_score = -2;      // p1 won, p2 lost
        else if (p1_score == -2) p2_score = 1; // p1 lost, p2 won

        //safely update both scores
        #pragma omp atomic
        scores[i] += p1_score;
        
        #pragma omp atomic
        scores[opp] += p2_score; //atomic:protects against thread collisions
      }
    }

    // selection: sort by score, keep top 15
    // create index array and sort by score descending
    std::vector<int> indices(combined.size());
    for (int i = 0; i < (int)indices.size(); i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    // build new population from top 15
    population.clear();
    for (int i = 0; i < POP_SIZE; i++) {
      population.push_back(combined[indices[i]]);
    }

    // print progress
    int best_score = scores[indices[0]];
    std::cout << "gen " << gen + 1 << "/" << NUM_GENERATIONS
              << std::endl;
  }

  std::cout << "\nevolution complete!" << std::endl;
  std::cout << "king value (fixed): " << KING_VALUE << std::endl;

  // save best network's weights to binary file
  NeuralNetwork &best = population[0];
  std::ofstream out("best_network.bin", std::ios::binary);
  if (out.is_open()) {
    // helper: write an Eigen matrix as raw doubles (row-major dump)
    auto write_matrix = [&](const Eigen::MatrixXd &m) {
      int rows = m.rows(), cols = m.cols();
      out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
      out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
      out.write(reinterpret_cast<const char*>(m.data()), rows * cols * sizeof(double));
    };



    // write weights and sigmas
    write_matrix(best.w1);
    write_matrix(best.w2);
    write_matrix(best.w3);
    write_matrix(best.sigma1);
    write_matrix(best.sigma2);
    write_matrix(best.sigma3);

    out.close();
    std::cout << "best network saved to best_network.bin" << std::endl;
  } else {
    std::cerr << "error: could not open best_network.bin for writing" << std::endl;
  }

  // save best network's weights to txt file
  std::ofstream txt("best_network.txt");
  if (txt.is_open()) {
    txt << std::fixed;
    txt.precision(8);

    auto write_matrix_txt = [&](const std::string &name, const Eigen::MatrixXd &m) {
      txt << name << " (" << m.rows() << "x" << m.cols() << "):\n";
      txt << m << "\n\n";
    };

    txt << "K (fixed): " << KING_VALUE << "\n\n";
    write_matrix_txt("w1", best.w1);
    write_matrix_txt("w2", best.w2);
    write_matrix_txt("w3", best.w3);
    write_matrix_txt("sigma1", best.sigma1);
    write_matrix_txt("sigma2", best.sigma2);
    write_matrix_txt("sigma3", best.sigma3);

    txt.close();
    std::cout << "best network saved to best_network.txt" << std::endl;
  } else {
    std::cerr << "error: could not open best_network.txt for writing" << std::endl;
  }

  return 0;
}
