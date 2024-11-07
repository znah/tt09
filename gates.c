#include <stdint.h>

#ifdef WASM
    #define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
    #define WASM_EXPORT(name)
#endif

#define BUFFER(name, type, size) type name[size]; \
  WASM_EXPORT("_get_"#name) type* get_##name() {return name;} \
  WASM_EXPORT("_len_"#name"__"#type) int get_##name##_len() {return (size);}


enum {
    MAX_GATE_N = 4096,
    MAX_IO_N = MAX_GATE_N*4,
    WAVE_STOP = 0xFFFF,
};

BUFFER(gate_n, uint32_t, 1)
BUFFER(luts, uint32_t, MAX_GATE_N);
BUFFER(inputs_start, uint32_t, MAX_GATE_N+1);
BUFFER(outputs_start, uint32_t, MAX_GATE_N+1);
BUFFER(inputs, uint32_t, MAX_IO_N);
BUFFER(outputs, uint32_t, MAX_IO_N);

BUFFER(state, uint8_t, MAX_GATE_N);
BUFFER(update_count, uint32_t, MAX_GATE_N);

BUFFER(queue, uint32_t, MAX_GATE_N);
BUFFER(queue_end, uint32_t, 2)
BUFFER(is_queued, uint8_t, MAX_GATE_N);

WASM_EXPORT("queue_len")
int queue_len() {
    return (queue_end[1]-queue_end[0]+MAX_GATE_N) % MAX_GATE_N;
}

void queue_push(uint32_t gate) {
    queue[queue_end[1]] = gate;
    queue_end[1] = (queue_end[1]+1) % MAX_GATE_N;
}

uint32_t queue_pop() {
    int i = queue_end[0];
    queue_end[0] = (i+1) % MAX_GATE_N;
    return queue[i];
}

uint8_t eval_gate(int gate) {
    int i0 = inputs_start[gate];
    int i1 = inputs_start[gate+1];
    uint32_t input_bits = 0;
    for (int i=i0,bit=0; i<i1; ++i,++bit) {
        input_bits += (state[inputs[i]]&1) << bit;
    }
    return (luts[gate] >> input_bits) & 1;
}

WASM_EXPORT("update_all") 
int update_all() {
    int count = 0;
    for (int i=0; i<gate_n[0]; ++i) {
        uint8_t v0 = state[i];
        uint8_t v = state[i] = eval_gate(i);
        count += v != v0;
    }
    return count;
}

WASM_EXPORT("set_signal")
int set_signal(int gate_i, uint8_t value) {
    if (value == state[gate_i]) {
        return queue_len();
    }
    state[gate_i] = value;
    update_count[gate_i]++;
    int i0 = outputs_start[gate_i];
    int i1 = outputs_start[gate_i+1];
    for (int i=i0; i<i1; ++i) {
        int gate = outputs[i];
        if (!is_queued[gate]) {
            is_queued[gate] = 1;
            queue_push(gate);
        }
    }
    return queue_len();
}

WASM_EXPORT("step")
int step() {
    if (queue_end[0] == queue_end[1]) {
        return 0;
    }
    int gate = queue_pop();
    is_queued[gate] = 0;
    return set_signal(gate, eval_gate(gate));
}

WASM_EXPORT("run_wave")
int run_wave() {
    int step_count = 0;
    queue_push(WAVE_STOP);
    while (queue[queue_end[0]] != WAVE_STOP) {
        step();
        step_count++;
    }
    queue_pop(); // remove WAVE_STOP
    return step_count;
}