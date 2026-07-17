/* kiTurbHIT3D — parameterized real-OpenLB HIT (DHIT + FHIT).
 *
 * KI-TURB agents control via case.xml:
 *   - TurbulenceRegime: dns | les
 *   - Collision: BGK/DNS, RLB, MRT, TRT, Smagorinsky, WALE,
 *     ConsistentStrainSmagorinsky, ShearSmagorinsky, Krause,
 *     SmagorinskyMRT, DynSmagorinsky
 *   - ForcingType: none, spectral_random, ornstein_uhlenbeck,
 *     constant_energy_input, constant_tke
 *   - InitialCondition: synthetic_spectrum, restart, imported_field
 *
 * Usage: kiTurbHIT3D <case.xml> <output_dir>
 */

#include "olb3D.h"
#ifndef OLB_PRECOMPILED
#include "olb3D.hh"
#endif

#include "dynamics/mrtDynamics.h"
#include "dynamics/mrtLatticeDescriptors.h"
#include "dynamics/WALELatticeDescriptors.h"
#include "dynamics/shearSmagorinskyLatticeDescriptors.h"
#include "dynamics/dynSmagorinskyLatticeDescriptors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <sys/stat.h>
#include <cerrno>
#include <string>
#include <type_traits>
#include <vector>

using namespace olb;
using namespace olb::descriptors;
using namespace olb::util;

typedef double T;

namespace {

// ---- strict case.xml helpers ---------------------------------------------
bool fileExists(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return static_cast<bool>(in);
}

std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string trim(std::string v) {
  const size_t s = v.find_first_not_of(" \t\r\n"), e = v.find_last_not_of(" \t\r\n");
  return (s == std::string::npos) ? std::string{} : v.substr(s, e - s + 1);
}

std::string tagValue(const std::string& xml, const std::string& tag, const std::string& fb = "") {
  const std::string open = "<" + tag + ">", close = "</" + tag + ">";
  auto a = xml.find(open);
  if (a == std::string::npos) return fb;
  a += open.size();
  auto b = xml.find(close, a);
  if (b == std::string::npos) throw std::runtime_error("missing closing tag for " + tag);
  return trim(xml.substr(a, b - a));
}

int tagInt(const std::string& xml, const std::string& tag, int fb) {
  const std::string v = tagValue(xml, tag);
  if (v.empty()) return fb;
  size_t n = 0; int out = std::stoi(v, &n);
  if (n != v.size()) throw std::runtime_error("invalid integer in <" + tag + ">: " + v);
  return out;
}

double tagDouble(const std::string& xml, const std::string& tag, double fb) {
  const std::string v = tagValue(xml, tag);
  if (v.empty()) return fb;
  size_t n = 0; double out = std::stod(v, &n);
  if (n != v.size() || !std::isfinite(out)) throw std::runtime_error("invalid number in <" + tag + ">: " + v);
  return out;
}

bool tagBool(const std::string& xml, const std::string& tag, bool fb) {
  std::string v = tagValue(xml, tag);
  if (v.empty()) return fb;
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c){ return std::tolower(c); });
  if (v == "true" || v == "1" || v == "yes") return true;
  if (v == "false" || v == "0" || v == "no") return false;
  throw std::runtime_error("invalid boolean in <" + tag + ">: " + v);
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
  return s;
}

std::string compact(std::string s) {
  s = lower(s);
  s.erase(std::remove_if(s.begin(), s.end(), [](char c){ return c=='_' || c=='-' || std::isspace(static_cast<unsigned char>(c)); }), s.end());
  return s;
}

std::string normColl(std::string s) {
  s = compact(s);
  if (s == "smagorinsky") return "smagorinskybgk";
  if (s == "constrain") return "consistentstrainsmagorinsky";
  if (s == "dynsmagorinsky") return "dynamicsmagorinsky";
  return s;
}

std::string normForce(std::string s) {
  s = lower(trim(s));
  std::replace(s.begin(), s.end(), '-', '_');
  if (s == "off") return "none";
  if (s == "spectral_low_k" || s == "low_wavenumber") return "spectral_random";
  if (s == "ou") return "ornstein_uhlenbeck";
  if (s == "linear" || s == "constant_energy") return "constant_energy_input";
  return s;
}

void rejectUnknownLeafTags(const std::string& xml) {
  static const std::set<std::string> allowed = {
    "SchemaVersion","Name","Flow","HITMode","TurbulenceRegime",
    "Lx","Ly","Lz","Nx","Ny","Nz","Lattice","Collision","Tau","Mach",
    "Viscosity","Reynolds","Density","CharLength","CharVelocity",
    "SmagorinskyConstant","TRTMagicParameter","InitialCondition","ICSeed",
    "ICKMin","ICKPeak","ICKMax","ICSpectrumModel","ICSpectrumExponent",
    "ICTargetUrms","TargetUrms","ICSourceFile","ICForcingStateFile","ICVerifyDivergenceTolerance",
    "ForcingType","ForcingPattern","ForcingKMin","ForcingKMax","ForcingAmplitude",
    "ForcingTargetInjectionRate","ForcingTargetTKE","ForcingCorrelationTime",
    "ForcingControllerGain","ForcingUpdateInterval","ForcingSeed","ForcingUnits",
    "RemoveMeanForce","SolenoidalProjection","TargetReLambda","MaxSteps",
    "OutputInterval","DiagnosticsInterval","CheckpointInterval","CheckpointDirectory",
    "CheckpointRetain","SampleStartStep","Format","WriteVelocity",
    "WritePressure","WriteDensity","WriteVorticity","WriteForcing","WritePopulations"
  };
  const std::regex leaf(R"(<([A-Za-z_][A-Za-z0-9_]*)>\s*[^<]*\s*</\1>)");
  for (std::sregex_iterator it(xml.begin(), xml.end(), leaf), stop; it != stop; ++it) {
    const std::string tag = (*it)[1].str();
    if (!allowed.count(tag)) throw std::runtime_error("unknown configuration field: " + tag);
  }
}

// ---- Fourier modes --------------------------------------------------------
struct FourierMode { T nx, ny, nz, ax, ay, az, phi; };

void projectSolenoidal(FourierMode& m) {
  const T k2 = m.nx*m.nx + m.ny*m.ny + m.nz*m.nz;
  if (k2 <= T()) { m.ax=m.ay=m.az=T(); return; }
  const T d = (m.nx*m.ax + m.ny*m.ay + m.nz*m.az) / k2;
  m.ax -= d*m.nx; m.ay -= d*m.ny; m.az -= d*m.nz;
}

std::vector<FourierMode> buildModes(int kmin, int kmax, unsigned seed,
                                    const std::string& spectrumModel, double kPeak,
                                    double spectrumExp, T& sumA2, double divergenceTolerance) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> uni(0.0, 1.0);
  std::normal_distribution<T> gauss(0.0, 1.0);
  std::vector<FourierMode> modes;
  sumA2 = 0.0;
  // Real cosine representation: one representative from each +/-k pair is
  // retained, which is equivalent to enforcing Hermitian symmetry.
  for (int kx = -kmax; kx <= kmax; ++kx)
    for (int ky = -kmax; ky <= kmax; ++ky)
      for (int kz = -kmax; kz <= kmax; ++kz) {
        if (kx < 0 || (kx == 0 && ky < 0) || (kx == 0 && ky == 0 && kz <= 0)) continue;
        T kmag = std::sqrt(T(kx*kx + ky*ky + kz*kz));
        if (kmag < kmin || kmag > kmax) continue;
        FourierMode m{T(kx),T(ky),T(kz),gauss(rng),gauss(rng),gauss(rng),T(2.0*M_PI*uni(rng))};
        projectSolenoidal(m);
        T amp = T();
        if (spectrumModel == "gaussian_k4") {
          const T kp = std::max<T>(T(kPeak), T(1e-12));
          // sqrt(E(k)) for E(k) proportional to k^4 exp[-2(k/kp)^2].
          amp = kmag*kmag*std::exp(-(kmag*kmag)/(kp*kp));
        } else if (spectrumModel == "von_karman_pao") {
          const T kp = std::max<T>(T(kPeak), T(1e-12));
          const T ratio = kmag/kp;
          amp = kmag*kmag/std::pow(T(1)+ratio*ratio,T(17.0/12.0));
        } else if (spectrumModel == "power_law") {
          amp = std::pow(kmag, spectrumExp);
        } else {
          throw std::runtime_error("unsupported spectrum model: "+spectrumModel);
        }
        m.ax*=amp; m.ay*=amp; m.az*=amp;
        const T residual=std::abs(m.nx*m.ax+m.ny*m.ay+m.nz*m.az);
        const T norm=std::sqrt((kmag*kmag)*(m.ax*m.ax+m.ay*m.ay+m.az*m.az))+T(1e-30);
        if (residual/norm>T(divergenceTolerance))
          throw std::runtime_error("synthetic Fourier mode failed solenoidal verification");
        sumA2 += m.ax*m.ax + m.ay*m.ay + m.az*m.az;
        modes.push_back(m);
      }
  return modes;
}

struct CaseParams {
  int Nx=64, Ny=64, Nz=64, maxSteps=2000, outInterval=500, diagnosticsInterval=100;
  int sampleStart=0, checkpointInterval=0, checkpointRetain=2;
  int icKmin=1, icKmax=4, forceKmin=1, forceKmax=2, forcingUpdateInterval=1;
  unsigned icSeed=12345, forcingSeed=23456;
  double Lx=2*M_PI, Ly=2*M_PI, Lz=2*M_PI, tau=0.51, mach=0.05;
  double charLength=2*M_PI, charPhysU=1.0, density=1.0, viscosity=-1.0, reynolds=100.0, icSpectrumExp=-2.0;
  double forcingAmplitude=0.1, forcingTargetInjectionRate=0.0, forcingTargetTKE=0.0;
  double forcingCorrelationTime=1.0, forcingControllerGain=0.1, smagoConst=0.1, trtMagic=0.25;
  double targetUrms=-1.0, targetReLambda=-1.0, icKpeak=4.0, verifyDivergenceTolerance=1e-8;
  bool forced=false, removeMeanForce=true, solenoidalProjection=true;
  bool writeVelocity=true, writePressure=false, writeDensity=true, writeVorticity=true;
  bool writeForcing=false, writePopulations=false;
  std::string caseName="openlb_hit", flow="hit", lattice="D3Q19", collision="smagorinskybgk", hitMode="decaying";
  std::string forcingType="none", forcingPattern="random_phase", forcingUnits="lattice_acceleration";
  std::string turbulenceRegime="les", initialCondition="synthetic_spectrum", icSpectrumModel="gaussian_k4", sourceFile, forcingStateFile;
  std::string checkpointDirectory="checkpoints", outputFormat="vtm", outDir;
};

CaseParams parseCase(const std::string& xml, const std::string& outDir) {
  rejectUnknownLeafTags(xml);
  CaseParams p; p.outDir=outDir;
  if (tagInt(xml,"SchemaVersion",2) != 2) throw std::runtime_error("only SchemaVersion 2 is supported");
  p.caseName=tagValue(xml,"Name","openlb_hit"); p.flow=lower(tagValue(xml,"Flow","hit"));
  p.Nx=tagInt(xml,"Nx",64); p.Ny=tagInt(xml,"Ny",p.Nx); p.Nz=tagInt(xml,"Nz",p.Nx);
  p.Lx=tagDouble(xml,"Lx",2*M_PI); p.Ly=tagDouble(xml,"Ly",p.Lx); p.Lz=tagDouble(xml,"Lz",p.Lx);
  p.lattice=tagValue(xml,"Lattice","D3Q19"); p.tau=tagDouble(xml,"Tau",0.51);
  p.mach=tagDouble(xml,"Mach",0.05); p.charLength=tagDouble(xml,"CharLength",p.Lx); p.charPhysU=tagDouble(xml,"CharVelocity",1.0);
  p.density=tagDouble(xml,"Density",1.0); p.viscosity=tagDouble(xml,"Viscosity",-1.0);
  p.reynolds=tagDouble(xml,"Reynolds",100.0); p.collision=normColl(tagValue(xml,"Collision","Smagorinsky"));
  p.turbulenceRegime=lower(tagValue(xml,"TurbulenceRegime","les"));
  p.hitMode=lower(tagValue(xml,"HITMode","decaying"));
  p.initialCondition=lower(tagValue(xml,"InitialCondition","synthetic_spectrum"));
  p.sourceFile=tagValue(xml,"ICSourceFile",""); p.forcingStateFile=tagValue(xml,"ICForcingStateFile","");
  p.forcingType=normForce(tagValue(xml,"ForcingType","none"));
  p.forcingPattern=lower(tagValue(xml,"ForcingPattern","random_phase"));
  p.icKmin=tagInt(xml,"ICKMin",1); p.icKpeak=tagDouble(xml,"ICKPeak",4.0); p.icKmax=tagInt(xml,"ICKMax",4);
  p.icSeed=static_cast<unsigned>(tagInt(xml,"ICSeed",12345));
  p.icSpectrumModel=lower(tagValue(xml,"ICSpectrumModel","gaussian_k4"));
  p.icSpectrumExp=tagDouble(xml,"ICSpectrumExponent",-2.0);
  p.verifyDivergenceTolerance=tagDouble(xml,"ICVerifyDivergenceTolerance",1e-8);
  p.targetUrms=tagDouble(xml,"ICTargetUrms",tagDouble(xml,"TargetUrms",-1.0));
  p.targetReLambda=tagDouble(xml,"TargetReLambda",-1.0);
  p.forceKmin=tagInt(xml,"ForcingKMin",1); p.forceKmax=tagInt(xml,"ForcingKMax",2);
  p.forcingAmplitude=tagDouble(xml,"ForcingAmplitude",0.1);
  p.forcingTargetInjectionRate=tagDouble(xml,"ForcingTargetInjectionRate",0.0);
  p.forcingTargetTKE=tagDouble(xml,"ForcingTargetTKE",0.0);
  p.forcingCorrelationTime=tagDouble(xml,"ForcingCorrelationTime",1.0);
  p.forcingControllerGain=tagDouble(xml,"ForcingControllerGain",0.1);
  p.forcingUpdateInterval=std::max(1,tagInt(xml,"ForcingUpdateInterval",1));
  p.forcingSeed=static_cast<unsigned>(tagInt(xml,"ForcingSeed",23456));
  p.forcingUnits=lower(tagValue(xml,"ForcingUnits","lattice_acceleration"));
  p.removeMeanForce=tagBool(xml,"RemoveMeanForce",true);
  p.solenoidalProjection=tagBool(xml,"SolenoidalProjection",true);
  p.maxSteps=tagInt(xml,"MaxSteps",2000); p.outInterval=std::max(1,tagInt(xml,"OutputInterval",500));
  p.diagnosticsInterval=std::max(1,tagInt(xml,"DiagnosticsInterval",100));
  p.checkpointInterval=std::max(0,tagInt(xml,"CheckpointInterval",0));
  p.checkpointDirectory=tagValue(xml,"CheckpointDirectory","checkpoints");
  p.checkpointRetain=std::max(1,tagInt(xml,"CheckpointRetain",2));
  p.sampleStart=tagInt(xml,"SampleStartStep",0);
  p.smagoConst=tagDouble(xml,"SmagorinskyConstant",0.1); p.trtMagic=tagDouble(xml,"TRTMagicParameter",0.25);
  p.outputFormat=lower(tagValue(xml,"Format","vtm"));
  p.writeVelocity=tagBool(xml,"WriteVelocity",true); p.writePressure=tagBool(xml,"WritePressure",false);
  p.writeDensity=tagBool(xml,"WriteDensity",true); p.writeVorticity=tagBool(xml,"WriteVorticity",true);
  p.writeForcing=tagBool(xml,"WriteForcing",false); p.writePopulations=tagBool(xml,"WritePopulations",false);
  p.forced=(p.hitMode=="forced") || p.forcingType!="none";
  if (p.viscosity <= 0.0) p.viscosity=p.charPhysU*p.Lx/p.reynolds;
  return p;
}

void validateCase(const CaseParams& p) {
  if (p.flow!="hit") throw std::runtime_error("Flow must be 'hit'");
  if (p.caseName.empty()) throw std::runtime_error("Name cannot be empty");
  if (p.Nx<4 || p.Ny<4 || p.Nz<4) throw std::runtime_error("Nx, Ny and Nz must be at least 4");
  if (p.Lx<=0 || p.Ly<=0 || p.Lz<=0 || p.viscosity<=0 || p.density<=0) throw std::runtime_error("domain lengths, density and viscosity must be positive");
  if (p.charLength<=0 || std::abs(p.charLength-p.Lx)>1e-12*std::max(1.0,std::abs(p.Lx))) throw std::runtime_error("CharLength must be positive and equal Lx in this versioned adapter");
  const double dx=p.Lx/p.Nx, dy=p.Ly/p.Ny, dz=p.Lz/p.Nz;
  if (std::max({dx,dy,dz})-std::min({dx,dy,dz}) > 1e-10*std::max({dx,dy,dz})) throw std::runtime_error("legacy OpenLB HIT requires uniform spacing Lx/Nx=Ly/Ny=Lz/Nz");
  if (compact(p.lattice)!="d3q19") throw std::runtime_error("this versioned adapter supports only D3Q19");
  if (p.tau<=0.5) throw std::runtime_error("Tau must be greater than 0.5");
  if (p.mach<=0 || p.mach>0.1) throw std::runtime_error("derived Mach must be in (0,0.1]");
  const int klim=std::min({p.Nx,p.Ny,p.Nz})/2-1;
  if (p.icKmin<1 || p.icKmax<p.icKmin || p.icKmax>klim) throw std::runtime_error("invalid initial-condition wavenumber band");
  if (p.forced && (p.forceKmin<1 || p.forceKmax<p.forceKmin || p.forceKmax>klim)) throw std::runtime_error("invalid forcing wavenumber band");
  static const std::set<std::string> forcing={"none","spectral_random","ornstein_uhlenbeck","constant_energy_input","constant_tke"};
  if (!forcing.count(p.forcingType)) throw std::runtime_error("unsupported forcing: "+p.forcingType);
  static const std::set<std::string> initial={"synthetic_spectrum","restart","imported_field"};
  if (!initial.count(p.initialCondition)) throw std::runtime_error("unsupported initial condition: "+p.initialCondition);
  static const std::set<std::string> spectra={"gaussian_k4","von_karman_pao","power_law"};
  if (p.initialCondition=="synthetic_spectrum" && !spectra.count(p.icSpectrumModel)) throw std::runtime_error("unsupported initial spectrum: "+p.icSpectrumModel);
  if (p.icKpeak<=0 || p.verifyDivergenceTolerance<=0) throw std::runtime_error("ICKPeak and ICVerifyDivergenceTolerance must be positive");
  if (p.initialCondition!="synthetic_spectrum" && p.sourceFile.empty()) throw std::runtime_error("restart/imported_field requires ICSourceFile");
  if (p.initialCondition!="synthetic_spectrum" && (p.sourceFile.size()<4 || lower(p.sourceFile.substr(p.sourceFile.size()-4))!=".khf")) throw std::runtime_error("restart/imported_field currently requires a KI-TURB .khf field checkpoint");
  if (p.initialCondition=="restart" && p.forced && p.forcingStateFile.empty()) throw std::runtime_error("forced restart requires ICForcingStateFile to preserve stochastic/controller state");
  if (p.outputFormat!="vti" && p.outputFormat!="vtm") throw std::runtime_error("legacy app supports only VTI/VTM output");
  if (p.writePopulations) throw std::runtime_error("population output is not available in the legacy OpenLB adapter; use field checkpoints");
  if (p.forcingType=="ornstein_uhlenbeck" && p.forcingCorrelationTime<=0) throw std::runtime_error("OU forcing requires positive correlation time");
  if (p.forcingType=="constant_energy_input" && p.forcingTargetInjectionRate<=0) throw std::runtime_error("constant_energy_input requires ForcingTargetInjectionRate");
  if (p.forcingType=="constant_tke" && p.forcingTargetTKE<=0) throw std::runtime_error("constant_tke requires ForcingTargetTKE");
  static const std::set<std::string> patterns={"random_phase","fixed_phase","sine","cosine","ou_process"};
  if (!patterns.count(p.forcingPattern)) throw std::runtime_error("unsupported forcing pattern: "+p.forcingPattern);
  if ((p.forcingType=="spectral_random" || p.forcingType=="ornstein_uhlenbeck") && !p.solenoidalProjection)
    throw std::runtime_error("spectral HIT forcing requires SolenoidalProjection=true");
  if (p.collision=="smagorinskymrt") throw std::runtime_error("SmagorinskyMRT is not implemented exactly in this legacy tree");
  if (p.forced && (p.collision=="rlb" || p.collision=="consistentstrainsmagorinsky" || p.collision=="krause" || p.collision=="dynamicsmagorinsky"))
    throw std::runtime_error("requested collision/forcing combination has no exact dynamics in this versioned adapter");
  static const std::set<std::string> collisions={"bgk","trt","mrt","rlb","smagorinskybgk","wale","shearsmagorinsky","consistentstrainsmagorinsky","krause","dynamicsmagorinsky"};
  if (!collisions.count(p.collision)) throw std::runtime_error("unsupported collision: "+p.collision);
}

std::string dynamicsClassName(const CaseParams& p) {
  if (p.collision=="wale") return p.forced?"WALEForcedBGKdynamics":"WALEBGKdynamics";
  if (p.collision=="shearsmagorinsky") return p.forced?"ShearSmagorinskyForcedBGKdynamics":"ShearSmagorinskyBGKdynamics";
  if (p.collision=="mrt") return p.forced?"ForcedMRTdynamics":"MRTdynamics";
  if (p.collision=="trt") return p.forced?"ForcedTRTdynamics":"TRTdynamics";
  if (p.collision=="rlb") return "RLBdynamics";
  if (p.collision=="smagorinskybgk") return p.forced?"SmagorinskyForcedBGKdynamics":"SmagorinskyBGKdynamics";
  if (p.collision=="consistentstrainsmagorinsky") return "ConStrainSmagorinskyBGKdynamics";
  if (p.collision=="krause") return "KrauseBGKdynamics";
  if (p.collision=="dynamicsmagorinsky") return "DynSmagorinskyBGKdynamics";
  return p.forced?"ForcedBGKdynamics":"BGKdynamics";
}

std::string effectiveJson(const CaseParams& p) {
  std::ostringstream o; o<<std::setprecision(17);
  o << "{\n  \"schema_version\": 2,\n  \"case_name\": \""<<p.caseName<<"\",\n  \"flow\": \""<<p.flow<<"\",\n  \"lattice\": \"D3Q19\",\n"
    << "  \"resolution\": ["<<p.Nx<<","<<p.Ny<<","<<p.Nz<<"],\n"
    << "  \"size\": ["<<p.Lx<<","<<p.Ly<<","<<p.Lz<<"],\n"
    << "  \"collision\": \""<<p.collision<<"\",\n"
    << "  \"dynamics_class\": \""<<dynamicsClassName(p)<<"\",\n"
    << "  \"forcing\": \""<<p.forcingType<<"\",\n"
    << "  \"initial_condition\": \""<<p.initialCondition<<"\",\n"
    << "  \"characteristic_length\": "<<p.charLength<<",\n"
    << "  \"characteristic_velocity\": "<<p.charPhysU<<",\n"
    << "  \"viscosity\": "<<p.viscosity<<",\n"
    << "  \"reynolds\": "<<p.reynolds<<",\n"
    << "  \"density\": "<<p.density<<",\n"
    << "  \"target_re_lambda\": "<<p.targetReLambda<<",\n"
    << "  \"output_format\": \""<<p.outputFormat<<"\",\n"
    << "  \"tau\": "<<p.tau<<",\n  \"mach\": "<<p.mach<<"\n}\n";
  return o.str();
}

void printCapabilities(std::ostream& o) {
  o << R"({"schema_version":2,"app":"kiTurbHIT3D","lattices":{"D3Q19":"supported","D3Q27":"unsupported"},"collision_models":{"BGK":"supported","TRT":"supported","MRT":"supported","RLB":"decaying_only","SmagorinskyBGK":"supported","WALE":"supported","ShearSmagorinsky":"supported","ConsistentStrainSmagorinsky":"decaying_only","Krause":"decaying_only","DynamicSmagorinsky":"decaying_only","SmagorinskyMRT":"unsupported"},"forcing":["none","spectral_random","ornstein_uhlenbeck","constant_energy_input","constant_tke"],"initial_conditions":{"synthetic_spectrum":"supported","restart":"field_level_khf","imported_field":"khf_only"},"output_formats":["vti","vtm"],"checkpoint":"field-level rho+velocity+step KHF; not population-exact","parallel_output":"rank-0 gathered VTI/VTM"})" << std::endl;
}

template <typename S>
class HITInitialField : public AnalyticalF3D<S, S> {
  const std::vector<FourierMode>& modes_;
  S kx_,ky_,kz_,scale_;
public:
  HITInitialField(const std::vector<FourierMode>& m, S lx,S ly,S lz,S scale)
      : AnalyticalF3D<S,S>(3), modes_(m), kx_(S(2*M_PI)/lx),ky_(S(2*M_PI)/ly),kz_(S(2*M_PI)/lz),scale_(scale) {}
  bool operator()(S o[], const S in[]) override {
    S ux=0,uy=0,uz=0;
    for (const auto& m:modes_) {
      S c=std::cos(kx_*m.nx*in[0]+ky_*m.ny*in[1]+kz_*m.nz*in[2]+m.phi);
      ux+=m.ax*c;uy+=m.ay*c;uz+=m.az*c;
    }
    o[0]=scale_*ux;o[1]=scale_*uy;o[2]=scale_*uz;return true;
  }
};


struct SyntheticFieldDiagnostics {
  double meanX=0,meanY=0,meanZ=0,urms=0,analyticDivergenceRms=0;
};

SyntheticFieldDiagnostics measureSyntheticField(const CaseParams& p,
                                                const std::vector<FourierMode>& modes,
                                                double scale) {
  SyntheticFieldDiagnostics d; const size_t n=static_cast<size_t>(p.Nx)*p.Ny*p.Nz;
  double sum2=0,div2=0; const double kx=2*M_PI/p.Lx,ky=2*M_PI/p.Ly,kz=2*M_PI/p.Lz;
  for(int z=0;z<p.Nz;++z)for(int y=0;y<p.Ny;++y)for(int x=0;x<p.Nx;++x){
    const double px=x*p.Lx/p.Nx,py=y*p.Ly/p.Ny,pz=z*p.Lz/p.Nz;double ux=0,uy=0,uz=0,div=0;
    for(const auto&m:modes){double phase=kx*m.nx*px+ky*m.ny*py+kz*m.nz*pz+m.phi;double c=std::cos(phase),sn=std::sin(phase);ux+=m.ax*c;uy+=m.ay*c;uz+=m.az*c;div-=sn*(kx*m.nx*m.ax+ky*m.ny*m.ay+kz*m.nz*m.az);}
    ux*=scale;uy*=scale;uz*=scale;div*=scale;d.meanX+=ux;d.meanY+=uy;d.meanZ+=uz;sum2+=ux*ux+uy*uy+uz*uz;div2+=div*div;
  }
  d.meanX/=n;d.meanY/=n;d.meanZ/=n;const double mean2=d.meanX*d.meanX+d.meanY*d.meanY+d.meanZ*d.meanZ;
  d.urms=std::sqrt(std::max(0.0,sum2/n-mean2)/3.0);d.analyticDivergenceRms=std::sqrt(div2/n);return d;
}

void writeInitialDiagnostics(const std::string& path,const CaseParams& p,
                             const SyntheticFieldDiagnostics& latticeDiagnostics,
                             double targetPhysical,double targetLattice,double measuredPhysical,double scale) {
  std::ofstream o(path);if(!o)throw std::runtime_error("cannot write initial-condition diagnostics");o<<std::setprecision(17)
    <<"{\n  \"type\": \"synthetic_spectrum\",\n  \"seed\": "<<p.icSeed<<",\n"
    <<"  \"velocity_units\": \"physical_and_lattice\",\n"
    <<"  \"target_urms_physical\": "<<targetPhysical<<",\n"
    <<"  \"target_urms_lattice\": "<<targetLattice<<",\n"
    <<"  \"measured_urms_physical\": "<<measuredPhysical<<",\n"
    <<"  \"measured_urms_lattice\": "<<latticeDiagnostics.urms<<",\n  \"scale\": "<<scale<<",\n"
    <<"  \"mean_velocity_lattice\": ["<<latticeDiagnostics.meanX<<","<<latticeDiagnostics.meanY<<","<<latticeDiagnostics.meanZ<<"],\n"
    <<"  \"analytic_divergence_rms_lattice\": "<<latticeDiagnostics.analyticDivergenceRms<<"\n}\n";
}

template <typename S>
class HITSpectralForceField : public AnalyticalF3D<S, S> {
  const std::vector<FourierMode>& modes_;
  S kx_,ky_,kz_,amp_;
  bool useCos_;
public:
  HITSpectralForceField(const std::vector<FourierMode>& m,S lx,S ly,S lz,S amp,bool useCos)
      : AnalyticalF3D<S,S>(3),modes_(m),kx_(S(2*M_PI)/lx),ky_(S(2*M_PI)/ly),kz_(S(2*M_PI)/lz),amp_(amp),useCos_(useCos) {}
  bool operator()(S o[],const S in[]) override {
    S fx=0,fy=0,fz=0;
    for (const auto& m:modes_) {S ph=kx_*m.nx*in[0]+ky_*m.ny*in[1]+kz_*m.nz*in[2]+m.phi;S v=useCos_?std::cos(ph):std::sin(ph);fx+=m.ax*v;fy+=m.ay*v;fz+=m.az*v;}
    o[0]=amp_*fx;o[1]=amp_*fy;o[2]=amp_*fz;return true;
  }
};

template <typename S>
class HITABCForceField : public AnalyticalF3D<S, S> {
  S kf_, amp_, phase_;
public:
  HITABCForceField(S box, S amp, S phase) : AnalyticalF3D<S,S>(3), kf_(S(2*M_PI)/box), amp_(amp), phase_(phase) {}
  bool operator()(S o[], const S in[]) override {
    S x = kf_*in[0], y = kf_*in[1], z = kf_*in[2];
    o[0] = amp_*(std::sin(y+phase_)+std::cos(z+phase_));
    o[1] = amp_*(std::sin(z+phase_)+std::cos(x+phase_));
    o[2] = amp_*(std::sin(x+phase_)+std::cos(y+phase_));
    return true;
  }
};

template <typename S, typename DESC>
class HITLinearForceField : public AnalyticalF3D<S, S> {
  AnalyticalFfromSuperF3D<S> interp_;
  S eps_, invMeanU2_;
public:
  HITLinearForceField(SuperLatticePhysVelocity3D<S,DESC>& vel, S eps, S meanU2)
      : AnalyticalF3D<S,S>(3), interp_(vel, true), eps_(eps),
        invMeanU2_(meanU2 > 0 ? S(1)/meanU2 : S(1)) {}
  bool operator()(S o[], const S in[]) override {
    S u[3]; interp_(u, in);
    o[0]=eps_*u[0]*invMeanU2_; o[1]=eps_*u[1]*invMeanU2_; o[2]=eps_*u[2]*invMeanU2_;
    return true;
  }
};

void writeVti(const std::string& path,int nx,int ny,int nz,double dx,double dy,double dz,
              const std::vector<double>& data,int nComp,const char* name) {
  std::ofstream out(path,std::ios::binary); if(!out) throw std::runtime_error("cannot write "+path);
  out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
  out << "  <ImageData WholeExtent=\"0 "<<nx-1<<" 0 "<<ny-1<<" 0 "<<nz-1<<"\" Origin=\"0 0 0\" Spacing=\""<<dx<<" "<<dy<<" "<<dz<<"\">\n";
  out << "    <Piece Extent=\"0 "<<nx-1<<" 0 "<<ny-1<<" 0 "<<nz-1<<"\">\n      <PointData>\n";
  out << "        <DataArray type=\"Float64\" Name=\""<<name<<"\" NumberOfComponents=\""<<nComp<<"\" format=\"appended\" offset=\"0\"/>\n";
  out << "      </PointData>\n    </Piece>\n  </ImageData>\n  <AppendedData encoding=\"raw\">\n_";
  std::uint32_t nbyte=static_cast<std::uint32_t>(data.size()*sizeof(double));out.write(reinterpret_cast<const char*>(&nbyte),sizeof(nbyte));out.write(reinterpret_cast<const char*>(data.data()),nbyte);out<<"\n  </AppendedData>\n</VTKFile>\n";
}

struct SampledFields {std::vector<double> rho,u;};
template <typename DESC>
SampledFields sampleFields(SuperLattice3D<T,DESC>& lat,UnitConverter<T,DESC> const& conv,const CaseParams& p) {
  SuperLatticePhysVelocity3D<T,DESC> vel(lat,conv); AnalyticalFfromSuperF3D<T> vi(vel,true);
  SuperLatticeDensity3D<T,DESC> den(lat); AnalyticalFfromSuperF3D<T> di(den,true);
  T dx=conv.getConversionFactorLength(),in[3],uo[3],ro[1];const size_t cells=static_cast<size_t>(p.Nx)*p.Ny*p.Nz;
  SampledFields f;f.rho.resize(cells);f.u.resize(cells*3);size_t q=0;
  for(int z=0;z<p.Nz;++z){in[2]=z*dx;for(int y=0;y<p.Ny;++y){in[1]=y*dx;for(int x=0;x<p.Nx;++x){in[0]=x*dx;vi(uo,in);di(ro,in);f.rho[q]=ro[0];f.u[3*q]=uo[0];f.u[3*q+1]=uo[1];f.u[3*q+2]=uo[2];++q;}}}
  return f;
}

std::vector<double> periodicVorticity(const std::vector<double>& u,const CaseParams& p,double dx,double dy,double dz) {
  std::vector<double> w(u.size(),0.0);auto id=[&](int x,int y,int z,int c){x=(x+p.Nx)%p.Nx;y=(y+p.Ny)%p.Ny;z=(z+p.Nz)%p.Nz;return 3*(static_cast<size_t>(z)*p.Ny*p.Nx+static_cast<size_t>(y)*p.Nx+x)+c;};
  for(int z=0;z<p.Nz;++z)for(int y=0;y<p.Ny;++y)for(int x=0;x<p.Nx;++x){size_t q=3*(static_cast<size_t>(z)*p.Ny*p.Nx+static_cast<size_t>(y)*p.Nx+x);double duy_dz=(u[id(x,y,z+1,1)]-u[id(x,y,z-1,1)])/(2*dz),duz_dy=(u[id(x,y+1,z,2)]-u[id(x,y-1,z,2)])/(2*dy);double duz_dx=(u[id(x+1,y,z,2)]-u[id(x-1,y,z,2)])/(2*dx),dux_dz=(u[id(x,y,z+1,0)]-u[id(x,y,z-1,0)])/(2*dz);double dux_dy=(u[id(x,y+1,z,0)]-u[id(x,y-1,z,0)])/(2*dy),duy_dx=(u[id(x+1,y,z,1)]-u[id(x-1,y,z,1)])/(2*dx);w[q]=duz_dy-duy_dz;w[q+1]=dux_dz-duz_dx;w[q+2]=duy_dx-dux_dy;}
  return w;
}


std::vector<double> computeForceField(const CaseParams& p,const std::vector<FourierMode>& modes,
                                      const SampledFields& fields,double dx,double latticeAccelScale) {
  const size_t n=fields.rho.size();std::vector<double> force(3*n,0.0);if(p.forcingType=="none")return force;
  double mx=0,my=0,mz=0;for(size_t i=0;i<n;++i){mx+=fields.u[3*i];my+=fields.u[3*i+1];mz+=fields.u[3*i+2];}mx/=n;my/=n;mz/=n;
  double meanU2=0;for(size_t i=0;i<n;++i){double a=fields.u[3*i]-mx,b=fields.u[3*i+1]-my,c=fields.u[3*i+2]-mz;meanU2+=a*a+b*b+c*c;}meanU2/=n;
  if(p.forcingType=="constant_energy_input"||p.forcingType=="constant_tke"){
    double coeff=p.forcingTargetInjectionRate;if(p.forcingType=="constant_tke")coeff=p.forcingControllerGain*(p.forcingTargetTKE-.5*meanU2);double inv=meanU2>0?1.0/meanU2:0.0;
    for(size_t i=0;i<n;++i){force[3*i]=coeff*(fields.u[3*i]-(p.removeMeanForce?mx:0))*inv;force[3*i+1]=coeff*(fields.u[3*i+1]-(p.removeMeanForce?my:0))*inv;force[3*i+2]=coeff*(fields.u[3*i+2]-(p.removeMeanForce?mz:0))*inv;}return force;
  }
  const double amp=p.forcingAmplitude*latticeAccelScale,kx=2*M_PI/p.Lx,ky=2*M_PI/p.Ly,kz=2*M_PI/p.Lz;size_t q=0;
  for(int z=0;z<p.Nz;++z)for(int y=0;y<p.Ny;++y)for(int x=0;x<p.Nx;++x){double fx=0,fy=0,fz=0;for(const auto&m:modes){double ph=kx*m.nx*(x*dx)+ky*m.ny*(y*dx)+kz*m.nz*(z*dx)+m.phi;double v=p.forcingPattern=="cosine"?std::cos(ph):std::sin(ph);fx+=m.ax*v;fy+=m.ay*v;fz+=m.az*v;}force[3*q]=amp*fx;force[3*q+1]=amp*fy;force[3*q+2]=amp*fz;++q;}
  if(p.removeMeanForce){double ax=0,ay=0,az=0;for(size_t i=0;i<n;++i){ax+=force[3*i];ay+=force[3*i+1];az+=force[3*i+2];}ax/=n;ay/=n;az/=n;for(size_t i=0;i<n;++i){force[3*i]-=ax;force[3*i+1]-=ay;force[3*i+2]-=az;}}
  return force;
}

struct FieldDiagnostics {double mass=0,rmin=0,rmax=0,umax=0,machMax=0,tke=0,dissipation=0,forcingPower=0,divergenceRms=0,reLambda=0,kmaxEta=0;};
FieldDiagnostics diagnoseFields(const CaseParams& p,const SampledFields& f,const std::vector<double>& force,double dx) {
  FieldDiagnostics d;const size_t n=f.rho.size();d.rmin=1e300;d.rmax=-1e300;double mx=0,my=0,mz=0;
  for(size_t i=0;i<n;++i){d.mass+=f.rho[i];d.rmin=std::min(d.rmin,f.rho[i]);d.rmax=std::max(d.rmax,f.rho[i]);mx+=f.u[3*i];my+=f.u[3*i+1];mz+=f.u[3*i+2];d.umax=std::max(d.umax,std::sqrt(f.u[3*i]*f.u[3*i]+f.u[3*i+1]*f.u[3*i+1]+f.u[3*i+2]*f.u[3*i+2]));}mx/=n;my/=n;mz/=n;
  auto id=[&](int x,int y,int z,int c){x=(x+p.Nx)%p.Nx;y=(y+p.Ny)%p.Ny;z=(z+p.Nz)%p.Nz;return 3*(static_cast<size_t>(z)*p.Ny*p.Nx+static_cast<size_t>(y)*p.Nx+x)+c;};double div2=0,s2=0,urms3=0,power=0;
  for(int z=0;z<p.Nz;++z)for(int y=0;y<p.Ny;++y)for(int x=0;x<p.Nx;++x){size_t q=static_cast<size_t>(z)*p.Ny*p.Nx+static_cast<size_t>(y)*p.Nx+x;double up[3]={f.u[3*q]-mx,f.u[3*q+1]-my,f.u[3*q+2]-mz};urms3+=up[0]*up[0]+up[1]*up[1]+up[2]*up[2];power+=f.u[3*q]*force[3*q]+f.u[3*q+1]*force[3*q+1]+f.u[3*q+2]*force[3*q+2];double g[3][3];for(int c=0;c<3;++c){g[c][0]=(f.u[id(x+1,y,z,c)]-f.u[id(x-1,y,z,c)])/(2*dx);g[c][1]=(f.u[id(x,y+1,z,c)]-f.u[id(x,y-1,z,c)])/(2*dx);g[c][2]=(f.u[id(x,y,z+1,c)]-f.u[id(x,y,z-1,c)])/(2*dx);}double div=g[0][0]+g[1][1]+g[2][2];div2+=div*div;for(int a=0;a<3;++a)for(int b=0;b<3;++b){double sij=.5*(g[a][b]+g[b][a]);s2+=sij*sij;}}
  d.tke=.5*urms3/n;d.dissipation=2*p.viscosity*s2/n;d.forcingPower=power/n;d.divergenceRms=std::sqrt(div2/n);d.machMax=d.umax/(p.charPhysU>0?p.charPhysU:1.0)*p.mach;double oneComp=urms3/(3*n);if(d.dissipation>0&&oneComp>0){double lambda=std::sqrt(15*p.viscosity*oneComp/d.dissipation);d.reLambda=std::sqrt(oneComp)*lambda/p.viscosity;double eta=std::pow(p.viscosity*p.viscosity*p.viscosity/d.dissipation,.25);d.kmaxEta=(M_PI/dx)*eta;}return d;
}

void ensureDirectory(const std::string& path){if(path.empty())return;if(::mkdir(path.c_str(),0775)!=0&&errno!=EEXIST)throw std::runtime_error("cannot create directory "+path);}
void writeForcingState(const std::string& path,const std::mt19937& rng,const std::vector<FourierMode>& modes){std::ofstream o(path);if(!o)throw std::runtime_error("cannot write forcing state "+path);o<<rng<<"\n"<<modes.size()<<"\n"<<std::setprecision(17);for(const auto&m:modes)o<<m.nx<<' '<<m.ny<<' '<<m.nz<<' '<<m.ax<<' '<<m.ay<<' '<<m.az<<' '<<m.phi<<"\n";}
void readForcingState(const std::string& path,std::mt19937& rng,std::vector<FourierMode>& modes){std::ifstream in(path);if(!in)throw std::runtime_error("cannot read forcing state "+path);size_t n;in>>rng>>n;modes.resize(n);for(auto&m:modes)in>>m.nx>>m.ny>>m.nz>>m.ax>>m.ay>>m.az>>m.phi;if(!in)throw std::runtime_error("invalid forcing state "+path);}

void writeCheckpointFile(const std::string& path,const CaseParams& p,int step,const SampledFields& f) {
  std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot write checkpoint "+path);
  const char magic[16]={'K','I','T','U','R','B','H','I','T','F','I','E','L','D','1','\0'};out.write(magic,16);std::int32_t dims[4]={p.Nx,p.Ny,p.Nz,step};out.write(reinterpret_cast<char*>(dims),sizeof(dims));
  for(size_t i=0;i<f.rho.size();++i){double row[4]={f.rho[i],f.u[3*i],f.u[3*i+1],f.u[3*i+2]};out.write(reinterpret_cast<char*>(row),sizeof(row));}
}

struct ImportedField {int nx=0,ny=0,nz=0,step=0;std::vector<double> rho,u;};
ImportedField readCheckpointFile(const std::string& path) {
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open imported/restart field: "+path);char magic[16];in.read(magic,16);if(std::string(magic)!="KITURBHITFIELD1")throw std::runtime_error("unsupported field checkpoint format");std::int32_t d[4];in.read(reinterpret_cast<char*>(d),sizeof(d));ImportedField f;f.nx=d[0];f.ny=d[1];f.nz=d[2];f.step=d[3];size_t n=static_cast<size_t>(f.nx)*f.ny*f.nz;f.rho.resize(n);f.u.resize(3*n);for(size_t i=0;i<n;++i){double row[4];in.read(reinterpret_cast<char*>(row),sizeof(row));if(!in)throw std::runtime_error("truncated field checkpoint");f.rho[i]=row[0];f.u[3*i]=row[1];f.u[3*i+1]=row[2];f.u[3*i+2]=row[3];}return f;
}

template <typename S> class ArrayVelocityField:public AnalyticalF3D<S,S>{const ImportedField& f_;S dx_,physicalToLattice_;public:ArrayVelocityField(const ImportedField& f,S dx,S physicalToLattice):AnalyticalF3D<S,S>(3),f_(f),dx_(dx),physicalToLattice_(physicalToLattice){}bool operator()(S o[],const S in[])override{int x=std::min(f_.nx-1,std::max(0,int(std::floor(in[0]/dx_+0.5)))),y=std::min(f_.ny-1,std::max(0,int(std::floor(in[1]/dx_+0.5)))),z=std::min(f_.nz-1,std::max(0,int(std::floor(in[2]/dx_+0.5))));size_t q=static_cast<size_t>(z)*f_.ny*f_.nx+static_cast<size_t>(y)*f_.nx+x;o[0]=physicalToLattice_*f_.u[3*q];o[1]=physicalToLattice_*f_.u[3*q+1];o[2]=physicalToLattice_*f_.u[3*q+2];return true;}};
template <typename S> class ArrayDensityField:public AnalyticalF3D<S,S>{const ImportedField& f_;S dx_;public:ArrayDensityField(const ImportedField& f,S dx):AnalyticalF3D<S,S>(1),f_(f),dx_(dx){}bool operator()(S o[],const S in[])override{int x=std::min(f_.nx-1,std::max(0,int(std::floor(in[0]/dx_+0.5)))),y=std::min(f_.ny-1,std::max(0,int(std::floor(in[1]/dx_+0.5)))),z=std::min(f_.nz-1,std::max(0,int(std::floor(in[2]/dx_+0.5))));size_t q=static_cast<size_t>(z)*f_.ny*f_.nx+static_cast<size_t>(y)*f_.nx+x;o[0]=f_.rho[q];return true;}};

void writeVtm(const CaseParams& p,int step,const std::vector<std::pair<std::string,std::string>>& files){if(p.outputFormat!="vtm")return;std::ostringstream n;n<<p.outDir<<"/fields_"<<step<<".vtm";std::ofstream o(n.str());o<<"<?xml version=\"1.0\"?>\n<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n  <vtkMultiBlockDataSet>\n";int i=0;for(auto&x:files)o<<"    <DataSet index=\""<<i++<<"\" name=\""<<x.first<<"\" file=\""<<x.second<<"\"/>\n";o<<"  </vtkMultiBlockDataSet>\n</VTKFile>\n";}

template <typename DESC>
T meanUSquared(SuperLattice3D<T,DESC>& lat,UnitConverter<T,DESC> const& conv,const CaseParams& p) {auto f=sampleFields(lat,conv,p);T mx=0,my=0,mz=0;size_t n=f.rho.size();for(size_t i=0;i<n;++i){mx+=f.u[3*i];my+=f.u[3*i+1];mz+=f.u[3*i+2];}mx/=n;my/=n;mz/=n;T s=0;for(size_t i=0;i<n;++i){T x=f.u[3*i]-mx,y=f.u[3*i+1]-my,z=f.u[3*i+2]-mz;s+=x*x+y*y+z*z;}return s/n;}

void advancePhases(std::vector<FourierMode>& modes, std::mt19937& rng) {
  std::uniform_real_distribution<T> uni(0.0, 2.0*M_PI);
  for (auto& m : modes) m.phi = uni(rng);
}

void ouStep(std::vector<FourierMode>& modes, std::mt19937& rng, T theta) {
  std::normal_distribution<T> gauss(0.0, 1.0);
  for (auto& m : modes) {
    m.ax = theta*m.ax + std::sqrt(1-theta*theta)*gauss(rng);
    m.ay = theta*m.ay + std::sqrt(1-theta*theta)*gauss(rng);
    m.az = theta*m.az + std::sqrt(1-theta*theta)*gauss(rng);
    projectSolenoidal(m);
  }
}

template <typename DESCRIPTOR>
typename std::enable_if<DESCRIPTOR::template provides<FORCE>(), void>::type
initZeroForce(SuperLattice3D<T,DESCRIPTOR>& lat, SuperGeometry3D<T>& geo) {
  AnalyticalConst3D<T,T> zf(std::vector<T>{T(0),T(0),T(0)});
  lat.template defineField<FORCE>(geo, 1, zf);
}

template <typename DESCRIPTOR>
typename std::enable_if<!DESCRIPTOR::template provides<FORCE>(), void>::type
initZeroForce(SuperLattice3D<T,DESCRIPTOR>&, SuperGeometry3D<T>&) {}

template <typename DESCRIPTOR>
typename std::enable_if<DESCRIPTOR::template provides<VELO_GRAD>(), void>::type
initZeroVeloGrad(SuperLattice3D<T,DESCRIPTOR>& lat, SuperGeometry3D<T>& geo) {
  AnalyticalConst3D<T,T> zg(std::vector<T>(DESCRIPTOR::template size<VELO_GRAD>(), T(0)));
  lat.template defineField<VELO_GRAD>(geo, 1, zg);
}

template <typename DESCRIPTOR>
typename std::enable_if<!DESCRIPTOR::template provides<VELO_GRAD>(), void>::type
initZeroVeloGrad(SuperLattice3D<T,DESCRIPTOR>&, SuperGeometry3D<T>&) {}

template <typename DESCRIPTOR>
typename std::enable_if<DESCRIPTOR::template provides<VELO_GRAD>(), void>::type
updateWaleVeloGrad(SuperLattice3D<T,DESCRIPTOR>& lat, SuperGeometry3D<T>& geo) {
  const int material = 1;
  if (geo.getStatistics().getNvoxel(material) == 0) return;

  auto sampleVel = [&](BlockLattice3D<T,DESCRIPTOR>& block, int iX, int iY, int iZ, T u[3]) {
    T rho;
    block.get(iX, iY, iZ).computeRhoU(rho, u);
  };

  for (int iC = 0; iC < lat.getLoadBalancer().size(); ++iC) {
    if (geo.getExtendedBlockGeometry(iC).getStatistics().getNvoxel(material) == 0) continue;
    auto& block = lat.getExtendedBlockLattice(iC);
    const int x0 = geo.getExtendedBlockGeometry(iC).getStatistics().getMinLatticeR(material)[0];
    const int y0 = geo.getExtendedBlockGeometry(iC).getStatistics().getMinLatticeR(material)[1];
    const int z0 = geo.getExtendedBlockGeometry(iC).getStatistics().getMinLatticeR(material)[2];
    const int x1 = geo.getExtendedBlockGeometry(iC).getStatistics().getMaxLatticeR(material)[0];
    const int y1 = geo.getExtendedBlockGeometry(iC).getStatistics().getMaxLatticeR(material)[1];
    const int z1 = geo.getExtendedBlockGeometry(iC).getStatistics().getMaxLatticeR(material)[2];

    for (int iX = x0 + 1; iX <= x1 - 1; ++iX) {
      for (int iY = y0 + 1; iY <= y1 - 1; ++iY) {
        for (int iZ = z0 + 1; iZ <= z1 - 1; ++iZ) {
          if (geo.getExtendedBlockGeometry(iC).getMaterial(iX, iY, iZ) != material) continue;

          T uXp[3], uXm[3], uYp[3], uYm[3], uZp[3], uZm[3];
          sampleVel(block, iX + 1, iY, iZ, uXp);
          sampleVel(block, iX - 1, iY, iZ, uXm);
          sampleVel(block, iX, iY + 1, iZ, uYp);
          sampleVel(block, iX, iY - 1, iZ, uYm);
          sampleVel(block, iX, iY, iZ + 1, uZp);
          sampleVel(block, iX, iY, iZ - 1, uZm);

          T grad[9] = {
            (uXp[0] - uXm[0]) * T(0.5), (uYp[0] - uYm[0]) * T(0.5), (uZp[0] - uZm[0]) * T(0.5),
            (uXp[1] - uXm[1]) * T(0.5), (uYp[1] - uYm[1]) * T(0.5), (uZp[1] - uZm[1]) * T(0.5),
            (uXp[2] - uXm[2]) * T(0.5), (uYp[2] - uYm[2]) * T(0.5), (uZp[2] - uZm[2]) * T(0.5),
          };
          block.get(iX, iY, iZ).template defineField<VELO_GRAD>(grad);
        }
      }
    }
  }
}

template <typename DESCRIPTOR>
typename std::enable_if<!DESCRIPTOR::template provides<VELO_GRAD>(), void>::type
updateWaleVeloGrad(SuperLattice3D<T,DESCRIPTOR>&, SuperGeometry3D<T>&) {}

template <typename DESCRIPTOR>
typename std::enable_if<DESCRIPTOR::template provides<SMAGO_CONST>(), void>::type
initSmagoConstField(SuperLattice3D<T,DESCRIPTOR>& lat, SuperGeometry3D<T>& geo, T cs) {
  AnalyticalConst3D<T,T> sc(cs);
  lat.template defineField<SMAGO_CONST>(geo, 1, sc);
}

template <typename DESCRIPTOR>
typename std::enable_if<!DESCRIPTOR::template provides<SMAGO_CONST>(), void>::type
initSmagoConstField(SuperLattice3D<T,DESCRIPTOR>&, SuperGeometry3D<T>&, T) {}

template <typename DESCRIPTOR>
typename std::enable_if<DESCRIPTOR::template provides<FORCE>(), void>::type
applyForcing(SuperLattice3D<T,DESCRIPTOR>& lat,SuperGeometry3D<T>& geo,
             UnitConverter<T,DESCRIPTOR> const& conv,const CaseParams& p,
             std::vector<FourierMode>& forceModes,std::mt19937& rng,int iT) {
  T amp=T(p.forcingAmplitude);
  if(p.forcingUnits=="physical_acceleration")amp*=conv.getConversionFactorTime()*conv.getConversionFactorTime()/conv.getConversionFactorLength();
  if(p.forcingType=="constant_energy_input" || p.forcingType=="constant_tke") {
    SuperLatticePhysVelocity3D<T,DESCRIPTOR> vel(lat,conv);T m2=meanUSquared<DESCRIPTOR>(lat,conv,p);T coefficient=T(p.forcingTargetInjectionRate);
    if(p.forcingType=="constant_tke")coefficient=T(p.forcingControllerGain)*(T(p.forcingTargetTKE)-T(.5)*m2);
    HITLinearForceField<T,DESCRIPTOR> f(vel,coefficient,m2);lat.template defineField<FORCE>(geo,1,f);return;
  }
  if(p.forcingType=="spectral_random" || p.forcingType=="ornstein_uhlenbeck") {
    if(p.forcingPattern=="random_phase")advancePhases(forceModes,rng);
    if(p.forcingType=="ornstein_uhlenbeck") {T theta=std::exp(-T(p.forcingUpdateInterval)*conv.getConversionFactorTime()/T(p.forcingCorrelationTime));ouStep(forceModes,rng,theta);}
    HITSpectralForceField<T> f(forceModes,T(p.Lx),T(p.Ly),T(p.Lz),amp,p.forcingPattern=="cosine");lat.template defineField<FORCE>(geo,1,f);return;
  }
  if(p.forcingType=="none") {AnalyticalConst3D<T,T> z(std::vector<T>{T(0),T(0),T(0)});lat.template defineField<FORCE>(geo,1,z);return;}
  throw std::runtime_error("unhandled forcing type "+p.forcingType);
}

template <typename DESCRIPTOR>
typename std::enable_if<!DESCRIPTOR::template provides<FORCE>(), void>::type
applyForcing(SuperLattice3D<T,DESCRIPTOR>&, SuperGeometry3D<T>&,
             UnitConverter<T,DESCRIPTOR> const&, const CaseParams&,
             std::vector<FourierMode>&, std::mt19937&, int) {}

template <typename DESCRIPTOR>
void runHITCore(const CaseParams& p,std::unique_ptr<Dynamics<T,DESCRIPTOR>> bulk) {
  singleton::directories().setOutputDir(p.outDir+"/");
  std::ofstream log(p.outDir+"/run.log");std::ofstream diagnostics(p.outDir+"/diagnostics.jsonl");
  log<<"kiTurbHIT3D case="<<p.caseName<<" collision="<<p.collision<<" dynamics="<<dynamicsClassName(p)<<" mode="<<(p.forced?"FHIT":"DHIT")<<" forcing="<<p.forcingType<<"\n";
  UnitConverterFromResolutionAndRelaxationTime<T,DESCRIPTOR> conv(int{p.Nx},T(p.tau),T(p.charLength),T(p.charPhysU),T(p.viscosity),T(p.density));conv.print();
#ifdef PARALLEL_MODE_MPI
  const int nc=singleton::mpi().getSize();const bool writer=singleton::mpi().getRank()==0;
#else
  const int nc=1;const bool writer=true;
#endif
  CuboidGeometry3D<T> cuboid(0,0,0,conv.getConversionFactorLength(),p.Nx,p.Ny,p.Nz,nc);cuboid.setPeriodicity(true,true,true);HeuristicLoadBalancer<T> lb(cuboid);SuperGeometry3D<T> geo(cuboid,lb,3);geo.rename(0,1);geo.communicate();
  SuperLattice3D<T,DESCRIPTOR> lat(geo);lat.defineDynamics(geo,0,&instances::getNoDynamics<T,DESCRIPTOR>());lat.defineDynamics(geo,1,bulk.get());lat.initialize();
  int startStep=0;
  if(p.initialCondition=="synthetic_spectrum") {T sumA2=0;auto modes=buildModes(p.icKmin,p.icKmax,p.icSeed,p.icSpectrumModel,p.icKpeak,p.icSpectrumExp,sumA2,p.verifyDivergenceTolerance);double targetPhysical=p.targetUrms>0?p.targetUrms:p.charPhysU;double targetLattice=conv.getLatticeVelocity(targetPhysical);auto raw=measureSyntheticField(p,modes,1.0);if(raw.urms<=0)throw std::runtime_error("synthetic field has zero RMS velocity");double scale=targetLattice/raw.urms;auto measured=measureSyntheticField(p,modes,scale);double measuredPhysical=conv.getPhysVelocity(measured.urms);if(std::abs(measured.urms-targetLattice)>1e-10*std::max(1.0,std::abs(targetLattice)))throw std::runtime_error("synthetic field RMS verification failed");if(measured.analyticDivergenceRms>p.verifyDivergenceTolerance)throw std::runtime_error("synthetic field divergence verification failed");if(writer)writeInitialDiagnostics(p.outDir+"/initial_condition_diagnostics.json",p,measured,targetPhysical,targetLattice,measuredPhysical,scale);AnalyticalConst3D<T,T> rho(T(p.density));HITInitialField<T> u0(modes,T(p.Lx),T(p.Ly),T(p.Lz),T(scale));lat.defineRhoU(geo,1,rho,u0);lat.iniEquilibrium(geo,1,rho,u0);}
  else {ImportedField field=readCheckpointFile(p.sourceFile);if(field.nx!=p.Nx||field.ny!=p.Ny||field.nz!=p.Nz)throw std::runtime_error("imported field dimensions do not match case");startStep=p.initialCondition=="restart"?field.step:0;if(startStep>p.maxSteps)throw std::runtime_error("restart checkpoint step exceeds MaxSteps");ArrayDensityField<T> rho(field,conv.getConversionFactorLength());ArrayVelocityField<T> u0(field,conv.getConversionFactorLength(),conv.getLatticeVelocity(T(1)));lat.defineRhoU(geo,1,rho,u0);lat.iniEquilibrium(geo,1,rho,u0);}
  if(p.forced){initZeroForce<DESCRIPTOR>(lat,geo);}
  initZeroVeloGrad<DESCRIPTOR>(lat,geo);initSmagoConstField<DESCRIPTOR>(lat,geo,T(p.smagoConst));lat.initialize();
  T fsum=0;std::vector<FourierMode> forceModes;std::mt19937 frng(p.forcingSeed);if(p.forced&&(p.forcingType=="spectral_random"||p.forcingType=="ornstein_uhlenbeck"))forceModes=buildModes(p.forceKmin,p.forceKmax,p.forcingSeed,"power_law",1.0,p.icSpectrumExp,fsum,p.verifyDivergenceTolerance);if(!p.forcingStateFile.empty())readForcingState(p.forcingStateFile,frng,forceModes);
  Timer<T> timer(std::max(1,p.maxSteps-startStep),geo.getStatistics().getNvoxel());timer.start();
  for(int iT=startStep;iT<=p.maxSteps;++iT){if(iT>0)updateWaleVeloGrad<DESCRIPTOR>(lat,geo);if(p.forced&&iT%p.forcingUpdateInterval==0)applyForcing<DESCRIPTOR>(lat,geo,conv,p,forceModes,frng,iT);
    const bool output=iT>=p.sampleStart&&iT%p.outInterval==0;const bool diag=iT%p.diagnosticsInterval==0;const bool manualChk=fileExists(p.outDir+"/checkpoint.request");const bool chk=manualChk||(p.checkpointInterval>0&&iT>0&&iT%p.checkpointInterval==0);
    if(output||diag||chk){auto f=sampleFields(lat,conv,p);if(writer){const double dx=conv.getConversionFactorLength();const double accelScale=p.forcingUnits=="physical_acceleration"?conv.getConversionFactorTime()*conv.getConversionFactorTime()/conv.getConversionFactorLength():1.0;auto force=computeForceField(p,forceModes,f,dx,accelScale);std::vector<std::pair<std::string,std::string>> files;auto write=[&](const std::string&var,const std::vector<double>&data,int comp){std::ostringstream base;base<<var<<"_"<<iT<<".vti";writeVti(p.outDir+"/"+base.str(),p.Nx,p.Ny,p.Nz,dx,dx,dx,data,comp,var.c_str());files.push_back({var,base.str()});};if(output){if(p.writeVelocity)write("velocity",f.u,3);if(p.writeDensity)write("density",f.rho,1);if(p.writeVorticity)write("vorticity",periodicVorticity(f.u,p,dx,dx,dx),3);if(p.writeForcing)write("forcing",force,3);if(p.writePressure){std::vector<double> pressure(f.rho.size());for(size_t j=0;j<pressure.size();++j)pressure[j]=(f.rho[j]-p.density)/3.0;write("pressure",pressure,1);}writeVtm(p,iT,files);}
      if(chk){const std::string cdir=p.outDir+"/"+p.checkpointDirectory;ensureDirectory(cdir);std::ostringstream cp;cp<<cdir<<"/checkpoint_"<<iT<<".khf";writeCheckpointFile(cp.str(),p,iT,f);std::ostringstream fs;fs<<cdir<<"/forcing_state_"<<iT<<".txt";writeForcingState(fs.str(),frng,forceModes);int old=p.checkpointInterval>0?iT-p.checkpointInterval*p.checkpointRetain:-1;if(old>0){std::ostringstream op,of;op<<cdir<<"/checkpoint_"<<old<<".khf";of<<cdir<<"/forcing_state_"<<old<<".txt";std::remove(op.str().c_str());std::remove(of.str().c_str());}if(manualChk)std::remove((p.outDir+"/checkpoint.request").c_str());}
      if(diag){auto d=diagnoseFields(p,f,force,dx);diagnostics<<"{\"step\":"<<iT<<",\"physical_time\":"<<conv.getPhysTime(iT)<<",\"progress\":"<<double(iT)/p.maxSteps<<",\"mass\":"<<d.mass<<",\"density_min\":"<<d.rmin<<",\"density_max\":"<<d.rmax<<",\"velocity_max\":"<<d.umax<<",\"mach_max\":"<<d.machMax<<",\"tke\":"<<d.tke<<",\"dissipation\":"<<d.dissipation<<",\"forcing_power\":"<<d.forcingPower<<",\"divergence_rms\":"<<d.divergenceRms<<",\"re_lambda\":"<<d.reLambda<<",\"kmax_eta\":"<<d.kmaxEta<<"}\n";diagnostics.flush();}
    }}lat.collideAndStream();}
  timer.stop();timer.printSummary();if(writer)log<<"done\n";
}

void dispatch(CaseParams& p) {
  if (p.collision == "wale") {
    if (p.forced) {
      UnitConverterFromResolutionAndRelaxationTime<T, WALEForcedD3Q19Descriptor> conv(
          int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
      T omega = conv.getLatticeRelaxationFrequency();
      auto bulk = std::unique_ptr<Dynamics<T, WALEForcedD3Q19Descriptor>>(
          new WALEForcedBGKdynamics<T, WALEForcedD3Q19Descriptor>(omega,
              instances::getBulkMomenta<T, WALEForcedD3Q19Descriptor>(), p.smagoConst));
      runHITCore<WALEForcedD3Q19Descriptor>(p, std::move(bulk));
    } else {
      UnitConverterFromResolutionAndRelaxationTime<T, WALED3Q19Descriptor> conv(
          int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
      T omega = conv.getLatticeRelaxationFrequency();
      auto bulk = std::unique_ptr<Dynamics<T, WALED3Q19Descriptor>>(
          new WALEBGKdynamics<T, WALED3Q19Descriptor>(
              omega, instances::getBulkMomenta<T, WALED3Q19Descriptor>(), p.smagoConst));
      runHITCore<WALED3Q19Descriptor>(p, std::move(bulk));
    }
    return;
  }

  if (p.collision == "shearsmagorinsky") {
    if (p.forced) {
      UnitConverterFromResolutionAndRelaxationTime<T, ShearSmagorinskyForcedD3Q19Descriptor> conv(
          int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
      T omega = conv.getLatticeRelaxationFrequency();
      auto bulk = std::unique_ptr<Dynamics<T, ShearSmagorinskyForcedD3Q19Descriptor>>(
          new ShearSmagorinskyForcedBGKdynamics<T, ShearSmagorinskyForcedD3Q19Descriptor>(
              omega, instances::getBulkMomenta<T, ShearSmagorinskyForcedD3Q19Descriptor>(), p.smagoConst));
      runHITCore<ShearSmagorinskyForcedD3Q19Descriptor>(p, std::move(bulk));
    } else {
      UnitConverterFromResolutionAndRelaxationTime<T, ShearSmagorinskyD3Q19Descriptor> conv(
          int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
      T omega = conv.getLatticeRelaxationFrequency();
      auto bulk = std::unique_ptr<Dynamics<T, ShearSmagorinskyD3Q19Descriptor>>(
          new ShearSmagorinskyBGKdynamics<T, ShearSmagorinskyD3Q19Descriptor>(
              omega, instances::getBulkMomenta<T, ShearSmagorinskyD3Q19Descriptor>(), p.smagoConst));
      runHITCore<ShearSmagorinskyD3Q19Descriptor>(p, std::move(bulk));
    }
    return;
  }

  if (p.collision == "mrt") {
    if (p.forced) {
      UnitConverterFromResolutionAndRelaxationTime<T, ForcedMRTD3Q19Descriptor> conv(
          int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
      T omega = conv.getLatticeRelaxationFrequency();
      auto bulk = std::unique_ptr<Dynamics<T, ForcedMRTD3Q19Descriptor>>(
          new ForcedMRTdynamics<T, ForcedMRTD3Q19Descriptor>(
              omega, instances::getBulkMomenta<T, ForcedMRTD3Q19Descriptor>()));
      runHITCore<ForcedMRTD3Q19Descriptor>(p, std::move(bulk));
    } else {
      UnitConverterFromResolutionAndRelaxationTime<T, MRTD3Q19Descriptor> conv(
          int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
      T omega = conv.getLatticeRelaxationFrequency();
      auto bulk = std::unique_ptr<Dynamics<T, MRTD3Q19Descriptor>>(
          new MRTdynamics<T, MRTD3Q19Descriptor>(
              omega, instances::getBulkMomenta<T, MRTD3Q19Descriptor>()));
      runHITCore<MRTD3Q19Descriptor>(p, std::move(bulk));
    }
    return;
  }

    if (p.collision == "dynamicsmagorinsky" && !p.forced) {
    UnitConverterFromResolutionAndRelaxationTime<T, DynSmagorinskyD3Q19Descriptor> conv(
        int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
    T omega = conv.getLatticeRelaxationFrequency();
    auto bulk = std::unique_ptr<Dynamics<T, DynSmagorinskyD3Q19Descriptor>>(
        new DynSmagorinskyBGKdynamics<T, DynSmagorinskyD3Q19Descriptor>(
            omega, instances::getBulkMomenta<T, DynSmagorinskyD3Q19Descriptor>()));
    runHITCore<DynSmagorinskyD3Q19Descriptor>(p, std::move(bulk));
    return;
  }

  // Standard D3Q19 / D3Q19<FORCE> family
  if (p.forced) {
    UnitConverterFromResolutionAndRelaxationTime<T, D3Q19<FORCE>> conv(
        int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
    T omega = conv.getLatticeRelaxationFrequency();
    auto& bm = instances::getBulkMomenta<T, D3Q19<FORCE>>();
    std::unique_ptr<Dynamics<T, D3Q19<FORCE>>> bulk;
    if (p.collision == "trt")
      bulk.reset(new ForcedTRTdynamics<T, D3Q19<FORCE>>(omega, bm, p.trtMagic));
    else if (p.collision == "smagorinskybgk")
      bulk.reset(new SmagorinskyForcedBGKdynamics<T, D3Q19<FORCE>>(omega, bm, p.smagoConst));
    else if (p.collision == "bgk")
      bulk.reset(new ForcedBGKdynamics<T, D3Q19<FORCE>>(omega, bm));
    else throw std::runtime_error("no exact forced dynamics for collision "+p.collision);
    runHITCore<D3Q19<FORCE>>(p, std::move(bulk));
  } else {
    UnitConverterFromResolutionAndRelaxationTime<T, D3Q19<>> conv(
        int{p.Nx}, T(p.tau), T(p.charLength), T(p.charPhysU), T(p.viscosity), T(p.density));
    T omega = conv.getLatticeRelaxationFrequency();
    auto& bm = instances::getBulkMomenta<T, D3Q19<>>();
    std::unique_ptr<Dynamics<T, D3Q19<>>> bulk;
    if (p.collision == "trt")
      bulk.reset(new TRTdynamics<T, D3Q19<>>(omega, bm, p.trtMagic));
    else if (p.collision == "rlb")
      bulk.reset(new RLBdynamics<T, D3Q19<>>(omega, bm));
    else if (p.collision == "consistentstrainsmagorinsky")
      bulk.reset(new ConStrainSmagorinskyBGKdynamics<T, D3Q19<>>(omega, bm, p.smagoConst));
    else if (p.collision == "krause")
      bulk.reset(new KrauseBGKdynamics<T, D3Q19<>>(omega, bm, p.smagoConst));
    else if (p.collision == "smagorinskybgk")
      bulk.reset(new SmagorinskyBGKdynamics<T, D3Q19<>>(omega, bm, p.smagoConst));
    else if (p.collision == "bgk")
      bulk.reset(new BGKdynamics<T, D3Q19<>>(omega, bm));
    else throw std::runtime_error("no exact decaying dynamics for collision "+p.collision);
    runHITCore<D3Q19<>>(p, std::move(bulk));
  }
}

}  // namespace

int main(int argc,char* argv[]) {
  olbInit(&argc,&argv);OstreamManager clout(std::cout,"kiTurbHIT3D");
  try {
    if(argc==2 && std::string(argv[1])=="--capabilities"){printCapabilities(std::cout);return 0;}
    if(argc>=3 && std::string(argv[1])=="--validate-only"){std::string xml=readFile(argv[2]);if(xml.empty())throw std::runtime_error("cannot read "+std::string(argv[2]));CaseParams p=parseCase(xml,".");validateCase(p);clout<<"configuration valid\n";return 0;}
    if(argc>=3 && std::string(argv[1])=="--dump-effective-config"){std::string xml=readFile(argv[2]);if(xml.empty())throw std::runtime_error("cannot read "+std::string(argv[2]));CaseParams p=parseCase(xml,".");validateCase(p);std::cout<<effectiveJson(p);return 0;}
    std::string casePath,outDir;
    if(argc>=4 && std::string(argv[1])=="--run"){casePath=argv[2];outDir=argv[3];}
    else if(argc>=3){casePath=argv[1];outDir=argv[2];}
    else {clout<<"usage: kiTurbHIT3D --capabilities | --validate-only case.xml | --dump-effective-config case.xml | --run case.xml output_dir\n";return 2;}
    std::string xml=readFile(casePath);if(xml.empty())throw std::runtime_error("cannot read "+casePath);CaseParams p=parseCase(xml,outDir);validateCase(p);ensureDirectory(outDir);std::ofstream(outDir+"/effective_openlb.json")<<effectiveJson(p);dispatch(p);clout<<"kiTurbHIT3D completed: "<<outDir<<std::endl;return 0;
  } catch(const std::exception& e){clout<<"error: "<<e.what()<<std::endl;return 1;}
}
