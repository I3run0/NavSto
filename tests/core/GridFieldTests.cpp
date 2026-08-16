#include "GridField.hpp"
#include "TestFramework.hpp"

TEST_CASE(GridField_ConstructsWithCorrectSize) {
    GridSize g{3, 4, 5};
    GridField<double> f(g, 1.5);
    REQUIRE(f.gridSize().sI == 3);
    REQUIRE(f.gridSize().sJ == 4);
    REQUIRE(f.gridSize().sK == 5);
    REQUIRE(f.data().size() == 3u * 4u * 5u);
    REQUIRE(f(0, 0, 0) == 1.5);
}

TEST_CASE(GridField_AccessorsReadWrite) {
    GridSize g{2, 2, 2};
    GridField<double> f(g, 0.0);
    f(1, 1, 1) = 42.0;
    REQUIRE(f(1, 1, 1) == 42.0);
    REQUIRE(f.at(0, 0, 0) == 0.0);
}

// The test binaries build with -UNDEBUG (see CMakeLists.txt) so this runs in
// every configuration. It used to be #ifndef NDEBUG'd and reported PASS in
// Release while checking nothing.
TEST_CASE(GridField_BoundsCheckThrows) {
    GridSize g{2, 2, 2};
    GridField<double> f(g);
    REQUIRE_THROWS(f.at(5, 0, 0));
    REQUIRE_THROWS(f.at(0, -1, 0));
}

TEST_CASE(GridField_ResizeReplacesStorage) {
    GridSize g1{2, 2, 2};
    GridField<double> f(g1, 7.0);
    GridSize g2{3, 3, 3};
    f.resize(g2, 0.0);
    REQUIRE(f.data().size() == 27u);
    REQUIRE(f(0, 0, 0) == 0.0);
}

