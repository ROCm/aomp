#include <chrono>

class Timer {
        std::chrono::time_point<std::chrono::steady_clock> _begin, _end;

    public:

        void begin() {
            _begin = std::chrono::steady_clock::now();
        }
        
        void end() {
            _end = std::chrono::steady_clock::now();
        }

        double elapsed() {
            return std::chrono::duration_cast<std::chrono::microseconds>(_end - _begin).count() / 1e6;  
        }
};

Timer t;

extern "C" {

    void start() {
        t.begin();
    }
    
    void stop() {
        t.end();
    }
    
    double elapsed() {
        return t.elapsed();
    }

}
