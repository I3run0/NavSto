#pragma once
// =============================================================================
//  TestFramework.hpp — minimal header-only unit-test harness.
//
//  Kept dependency-free (no Catch2/GTest) to match the project's "no
//  external libraries" build philosophy. Tests self-register via static
//  initializers; main.cpp just calls testfw::runAll().
// =============================================================================

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace testfw {

struct Test {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Test>& registry() {
    static std::vector<Test> tests;
    return tests;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

struct AssertionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline int runAll() {
    int failed = 0;
    for (auto& t : registry()) {
        try {
            t.fn();
            std::cout << "[ PASS ] " << t.name << "\n";
        } catch (const std::exception& e) {
            std::cout << "[ FAIL ] " << t.name << " -- " << e.what() << "\n";
            ++failed;
        }
    }
    std::cout << "\n" << (registry().size() - static_cast<std::size_t>(failed))
               << "/" << registry().size() << " tests passed.\n";
    return failed;
}

} // namespace testfw

#define TEST_CASE(name)                                                      \
    static void name();                                                     \
    static ::testfw::Registrar registrar_##name(#name, name);                \
    static void name()

#define REQUIRE(cond)                                                        \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::ostringstream oss_;                                         \
            oss_ << "REQUIRE(" #cond ") failed at " << __FILE__ << ":"       \
                 << __LINE__;                                                \
            throw ::testfw::AssertionFailure(oss_.str());                    \
        }                                                                    \
    } while (0)

#define REQUIRE_THROWS(expr)                                                 \
    do {                                                                     \
        bool threw_ = false;                                                 \
        try {                                                                \
            (void)(expr);                                                   \
        } catch (...) {                                                      \
            threw_ = true;                                                   \
        }                                                                    \
        if (!threw_) {                                                       \
            std::ostringstream oss_;                                         \
            oss_ << "REQUIRE_THROWS(" #expr ") did not throw at " << __FILE__ \
                 << ":" << __LINE__;                                         \
            throw ::testfw::AssertionFailure(oss_.str());                    \
        }                                                                    \
    } while (0)
