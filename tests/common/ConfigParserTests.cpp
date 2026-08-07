#include "ConfigParser.hpp"
#include "TestFramework.hpp"

#include <filesystem>
#include <fstream>

namespace {

std::filesystem::path writeTempConfig(const std::string& contents) {
    auto path = std::filesystem::temp_directory_path() / "navsolver_test_config.cfg";
    std::ofstream f(path);
    f << contents;
    return path;
}

} // namespace

TEST_CASE(ConfigParser_ParsesKnownKeys) {
    auto path = writeTempConfig(
        "# comment line\n"
        "numCellsX = 32\n"
        "numCellsY = 16\n"
        "numCellsZ = 8\n"
        "reynoldsNumber = 250.5\n"
        "geometryShape = AbruptExpansion\n"
        "lateralBC = SolidWall\n"
        "outletBC = ZeroSecondDeriv\n"
        "initialProfile = InletProfile\n"
        "flowType = RK4Transient\n"
        "outputDir = my_results\n"
        "runName = unit_test_run\n");

    SimConfig cfg;
    ConfigParser::parse(path, cfg);

    REQUIRE(cfg.numCellsX == 32);
    REQUIRE(cfg.numCellsY == 16);
    REQUIRE(cfg.numCellsZ == 8);
    REQUIRE(cfg.reynoldsNumber == 250.5);
    REQUIRE(cfg.geometryShape == GeometryShape::AbruptExpansion);
    REQUIRE(cfg.lateralCondition == LateralBC::SolidWall);
    REQUIRE(cfg.outletCondition == OutletBC::ZeroSecondDeriv);
    REQUIRE(cfg.initialProfile == InitialProfile::InletProfile);
    REQUIRE(cfg.flowType == FlowType::RK4Transient);
    REQUIRE(cfg.outputDir == "my_results");
    REQUIRE(cfg.runName == "unit_test_run");

    std::filesystem::remove(path);
}

TEST_CASE(ConfigParser_DerivesCellSizeFromDomainAndGrid) {
    auto path = writeTempConfig(
        "numCellsX = 10\n"
        "domainLengthX = 5.0\n");

    SimConfig cfg;
    ConfigParser::parse(path, cfg);

    REQUIRE(cfg.cellSizeX == 0.5);

    std::filesystem::remove(path);
}

TEST_CASE(ConfigParser_UnknownGeometryShapeThrows) {
    auto path = writeTempConfig("geometryShape = NotARealShape\n");

    SimConfig cfg;
    REQUIRE_THROWS(ConfigParser::parse(path, cfg));

    std::filesystem::remove(path);
}

TEST_CASE(ConfigParser_MissingFileThrows) {
    SimConfig cfg;
    REQUIRE_THROWS(ConfigParser::parse("/no/such/file/here.cfg", cfg));
}

TEST_CASE(ConfigParser_MalformedLineThrows) {
    auto path = writeTempConfig("this line has no equals sign\n");

    SimConfig cfg;
    REQUIRE_THROWS(ConfigParser::parse(path, cfg));

    std::filesystem::remove(path);
}
