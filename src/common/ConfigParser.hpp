#pragma once
// =============================================================================
//  ConfigParser.hpp  —  Reads a simple key = value config file into SimConfig.
//
//  Format (lines starting with '#' are comments):
//
//    numCellsX      = 48
//    numCellsY      = 24
//    numCellsZ      = 24
//    baseUnit       = 12
//    reynoldsNumber = 100.0
//    geometryShape  = AbruptExpansion
//    lateralBC      = Periodic
//    outletBC       = ZeroFirstDeriv
//    initialProfile = InletProfile
//    flowType       = RK4Transient
//    maxTimeSteps   = 10000
//    reportEveryN   = 500
//    convergenceTol = 1e-6
//    outputDir      = ./results
//    runName        = channel_re100
// =============================================================================

#include "SimState.hpp"
#include "Logger.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

class ConfigParser {
public:
    /// Parse the file at `path` and fill `cfg`.  Throws on unknown keys or
    /// malformed values so configuration errors surface before any computation.
    static void parse(const std::filesystem::path& path, SimConfig& cfg) {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("Cannot open config file: " + path.string());

        std::unordered_map<std::string, std::string> kv;
        std::string line;
        int lineNo = 0;
        while (std::getline(f, line)) {
            ++lineNo;
            // Strip comments and whitespace
            auto commentPos = line.find('#');
            if (commentPos != std::string::npos) line = line.substr(0, commentPos);
            if (trim(line).empty()) continue;

            auto eq = line.find('=');
            if (eq == std::string::npos)
                throw std::runtime_error("Config line " + std::to_string(lineNo)
                                         + ": missing '=' in \"" + line + "\"");

            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            kv[key] = val;
        }

        // Apply parsed values to cfg
        setIfPresent(kv, "numCellsX",        cfg.numCellsX);
        setIfPresent(kv, "numCellsY",        cfg.numCellsY);
        setIfPresent(kv, "numCellsZ",        cfg.numCellsZ);
        setIfPresent(kv, "baseUnit",         cfg.baseUnit);
        setIfPresent(kv, "domainLengthX",    cfg.domainLengthX);
        setIfPresent(kv, "domainLengthY",    cfg.domainLengthY);
        setIfPresent(kv, "domainLengthZ",    cfg.domainLengthZ);
        setIfPresent(kv, "reynoldsNumber",   cfg.reynoldsNumber);
        setIfPresent(kv, "hyperViscousRe",   cfg.hyperViscousRe);
        setIfPresent(kv, "hyperViscousStart",cfg.hyperViscousStart);
        setIfPresent(kv, "maxTimeSteps",     cfg.maxTimeSteps);
        setIfPresent(kv, "reportEveryN",     cfg.reportEveryN);
        setIfPresent(kv, "convergenceTol",   cfg.convergenceTol);
        setIfPresent(kv, "numPressureIter",  cfg.numPressureIter);

        if (kv.count("outputDir"))  cfg.outputDir = kv["outputDir"];
        if (kv.count("runName"))    cfg.runName   = kv["runName"];

        if (kv.count("geometryShape"))  cfg.geometryShape   = parseShape(kv["geometryShape"]);
        if (kv.count("geometryType"))   cfg.geometryType    = parseGeoType(kv["geometryType"]);
        if (kv.count("lateralBC"))      cfg.lateralCondition = parseLateral(kv["lateralBC"]);
        if (kv.count("outletBC"))       cfg.outletCondition  = parseOutlet(kv["outletBC"]);
        if (kv.count("initialProfile")) cfg.initialProfile   = parseProfile(kv["initialProfile"]);
        if (kv.count("flowType"))       cfg.flowType         = parseFlow(kv["flowType"]);

        // Cell sizes: dx = Cmp/II, dy = Alt/JJ, dz = Lrg/KK  (matches NavSto_dynamic.cpp)
        if (cfg.cellSizeX == 0.0) cfg.cellSizeX = cfg.domainLengthX / cfg.numCellsX;
        if (cfg.cellSizeY == 0.0) cfg.cellSizeY = cfg.domainLengthY / cfg.numCellsY;
        if (cfg.cellSizeZ == 0.0) cfg.cellSizeZ = cfg.domainLengthZ / cfg.numCellsZ;

        LOG_INFO("Config loaded from ", path.string());
    }

    /// Write current config back to a file (useful for checkpointing / provenance).
    static void write(const std::filesystem::path& path, const SimConfig& cfg) {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot write config: " + path.string());
        f << "# NavSolver configuration snapshot\n";
        f << "numCellsX        = " << cfg.numCellsX        << '\n';
        f << "numCellsY        = " << cfg.numCellsY        << '\n';
        f << "numCellsZ        = " << cfg.numCellsZ        << '\n';
        f << "baseUnit         = " << cfg.baseUnit         << '\n';
        f << "domainLengthX    = " << cfg.domainLengthX    << '\n';
        f << "domainLengthY    = " << cfg.domainLengthY    << '\n';
        f << "domainLengthZ    = " << cfg.domainLengthZ    << '\n';
        f << "reynoldsNumber   = " << cfg.reynoldsNumber   << '\n';
        f << "hyperViscousRe   = " << cfg.hyperViscousRe   << '\n';
        f << "maxTimeSteps     = " << cfg.maxTimeSteps     << '\n';
        f << "reportEveryN     = " << cfg.reportEveryN     << '\n';
        f << "convergenceTol   = " << cfg.convergenceTol   << '\n';
        f << "numPressureIter  = " << cfg.numPressureIter  << '\n';
        f << "outputDir        = " << cfg.outputDir.string() << '\n';
        f << "runName          = " << cfg.runName          << '\n';
    }

private:
    static std::string trim(std::string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c){ return !std::isspace(c); }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
        return s;
    }

    // Integer setter
    static void setIfPresent(const std::unordered_map<std::string,std::string>& kv,
                              const std::string& key, int& out) {
        auto it = kv.find(key);
        if (it != kv.end()) out = std::stoi(it->second);
    }
    // Double setter
    static void setIfPresent(const std::unordered_map<std::string,std::string>& kv,
                              const std::string& key, double& out) {
        auto it = kv.find(key);
        if (it != kv.end()) out = std::stod(it->second);
    }

    static GeometryShape parseShape(const std::string& v) {
        if (v == "AbruptExpansion")       return GeometryShape::AbruptExpansion;
        if (v == "AbruptContraction")     return GeometryShape::AbruptContraction;
        if (v == "OpenCavity")            return GeometryShape::OpenCavity;
        if (v == "GradualExpansion")      return GeometryShape::GradualExpansion;
        if (v == "UnilateralExpansion")   return GeometryShape::UnilateralExpansion;
        if (v == "GradualContraction")    return GeometryShape::GradualContraction;
        if (v == "UnilateralContraction") return GeometryShape::UnilateralContraction;
        if (v == "SharpCorner")           return GeometryShape::SharpCorner;
        if (v == "RoundedCorner")         return GeometryShape::RoundedCorner;
        throw std::runtime_error("Unknown geometryShape: " + v);
    }
    static GeometryType parseGeoType(const std::string& v) {
        if (v == "Axial")  return GeometryType::Axial;
        if (v == "Curved") return GeometryType::Curved;
        throw std::runtime_error("Unknown geometryType: " + v);
    }
    static LateralBC parseLateral(const std::string& v) {
        if (v == "Periodic")  return LateralBC::Periodic;
        if (v == "SolidWall") return LateralBC::SolidWall;
        throw std::runtime_error("Unknown lateralBC: " + v);
    }
    static OutletBC parseOutlet(const std::string& v) {
        if (v == "ZeroFirstDeriv")  return OutletBC::ZeroFirstDeriv;
        if (v == "ZeroSecondDeriv") return OutletBC::ZeroSecondDeriv;
        throw std::runtime_error("Unknown outletBC: " + v);
    }
    static InitialProfile parseProfile(const std::string& v) {
        if (v == "InletProfile")  return InitialProfile::InletProfile;
        if (v == "PotentialFlow") return InitialProfile::PotentialFlow;
        throw std::runtime_error("Unknown initialProfile: " + v);
    }
    static FlowType parseFlow(const std::string& v) {
        if (v == "SteadyMarching") return FlowType::SteadyMarching;
        if (v == "RK4Transient")   return FlowType::RK4Transient;
        throw std::runtime_error("Unknown flowType: " + v);
    }
};