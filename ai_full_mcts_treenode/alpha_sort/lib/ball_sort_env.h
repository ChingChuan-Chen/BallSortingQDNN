#ifndef BALL_SORT_ENV_H
#define BALL_SORT_ENV_H
#include <vector>
#include <tuple>

class BallSortEnv {
public:
    // Constructor with state
    BallSortEnv(int num_colors, int tube_capacity, int num_empty_tubes, const std::vector<std::vector<int8_t>>& state);

    // Constructor without state (random initialization)
    BallSortEnv(int num_colors, int tube_capacity, int num_empty_tubes);

    void reset();
    int top_index(int tube_idx) const;
    int get_top_color_streak(int tube_idx) const;
    bool is_full_tube(int tube_idx) const;
    bool is_empty_tube(int tube_idx) const;
    bool is_completed_tube(int tube_idx) const;
    bool is_valid_move(int src, int dst) const;
    std::vector<std::pair<int, int>> get_valid_actions();
    bool move(int src, int dst);
    void undo_move(int src, int dst);
    std::vector<std::vector<std::vector<int8_t>>> get_encoded_state(
        int max_num_colors,
        int num_empty_tubes,
        int max_tube_capacity
    );
    bool have_valid_moves() const;

    // Getters and Setters
    std::vector<std::vector<int8_t>> get_state() const { return state; }
    void set_move_count(int move_count) { this->move_count = move_count; }
    void set_is_done(bool is_done) { this->is_done = is_done; }
    bool get_is_done() const { return is_done; }
    bool get_is_moved() const { return is_moved; }
    bool get_is_solved();
    int get_num_colors() const { return num_colors; }
    int get_tube_capacity() const { return tube_capacity; }
    int get_num_empty_tubes() const { return num_empty_tubes; }
    int get_num_tubes() const { return num_tubes; }
    int get_move_count() const { return move_count; }

private:
    int num_colors;
    int tube_capacity;
    int num_empty_tubes;
    int num_tubes;
    int move_count;
    bool is_done;
    bool is_moved;
    bool is_solved;
    std::vector<std::vector<int8_t>> state;

    // check if the state is valid
    bool is_valid_state() const;

    // Cache for is_valid_move
    std::vector<std::vector<bool>> valid_move_cache;

    // Helper function to update the cache
    void update_valid_move_cache();
    bool check_is_valid_move(int src, int dst) const;
};

#endif
