"""Strict translation from canonical KI-TURB HIT configuration to OpenLB XML."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any,Dict
from xml.etree.ElementTree import Element,ElementTree,SubElement,indent
from schemas.openlb_hit import OpenLBHITConfig
from .capability_validator import OpenLBHITCapabilityValidator
class OpenLBHITConfigTranslator:
    def __init__(self,capability_validator=None):self.capability_validator=capability_validator or OpenLBHITCapabilityValidator()
    def effective_configuration(self,config):
        decision=self.capability_validator.assert_supported(config); derived=config.derive_scaling()
        return {"schema_version":2,"requested":config.model_dump(mode="json"),"derived_scaling":derived.model_dump(mode="json"),"capability_decision":decision.model_dump(mode="json"),"requested_equals_effective":True,"runtime_verified":False,
                "collision":{"model":config.collision.model.value},"forcing":{"type":config.forcing.type.value},
                "effective_openlb":{"lattice":config.domain.lattice.value,"collision":config.collision.model.value,"forcing":config.forcing.type.value,"initial_condition":config.initial_condition.type.value}}
    @staticmethod
    def _tag(parent,name,value):
        child=SubElement(parent,name);child.text=str(value)
    @classmethod
    def _optional(cls,parent,name,value):
        if value is not None:cls._tag(parent,name,value)
    def render_xml(self,config:OpenLBHITConfig)->str:
        eff=self.effective_configuration(config);d=eff["derived_scaling"]
        root=Element("Param");self._tag(root,"SchemaVersion",2)
        case=SubElement(root,"Case");self._tag(case,"Name",config.name);self._tag(case,"Flow","hit");self._tag(case,"HITMode","decaying" if config.forcing.type.value=="none" else "forced");self._tag(case,"TurbulenceRegime",config.metadata.get("turbulence_regime","dns" if config.collision.model.value in {"BGK","TRT","MRT","RLB"} else "les"))
        geo=SubElement(root,"Geometry")
        for a,v in zip("xyz",config.domain.size):self._tag(geo,"L"+a,v)
        mesh=SubElement(root,"Mesh")
        for a,v in zip("xyz",config.domain.resolution):self._tag(mesh,"N"+a,v)
        lbm=SubElement(root,"LBM");
        for k,v in [("Lattice",config.domain.lattice.value),("Collision",("Smagorinsky" if config.collision.model.value=="SmagorinskyBGK" else config.collision.model.value)),("Tau",d["relaxation_time"]),("Mach",d["actual_mach"]),("Viscosity",d["physical_viscosity"]),("Reynolds",d["reynolds_number"]),("Density",config.scaling.density),("CharLength",d["characteristic_length"]),("CharVelocity",d["characteristic_velocity"])]:self._tag(lbm,k,v)
        self._optional(lbm,"SmagorinskyConstant",config.collision.smagorinsky_constant);self._optional(lbm,"TRTMagicParameter",config.collision.trt_magic_parameter)
        hit=SubElement(root,"HIT");ic=config.initial_condition;f=config.forcing
        for k,v in [("InitialCondition",ic.type.value),("ICSeed",ic.seed),("ICKMin",ic.wavenumber_min),("ICKPeak",ic.wavenumber_peak),("ICKMax",ic.wavenumber_max),("ICSpectrumModel",ic.spectrum_model),("ICSpectrumExponent",ic.spectrum_exponent),("ICTargetUrms",ic.target_urms),("ICSourceFile",ic.source_file),("ICForcingStateFile",ic.forcing_state_file),("ICVerifyDivergenceTolerance",ic.verify_divergence_tolerance),("ForcingType",f.type.value),("ForcingPattern",config.metadata.get("forcing_pattern_legacy","random_phase")),("ForcingKMin",f.wavenumber_min),("ForcingKMax",f.wavenumber_max),("ForcingAmplitude",f.amplitude),("ForcingTargetInjectionRate",f.target_injection_rate),("ForcingTargetTKE",f.target_tke),("ForcingCorrelationTime",f.correlation_time),("ForcingControllerGain",f.controller_gain),("ForcingUpdateInterval",f.update_interval),("ForcingSeed",f.seed),("ForcingUnits",f.units),("RemoveMeanForce",str(f.remove_mean_force).lower()),("SolenoidalProjection",str(f.solenoidal_projection).lower()),("TargetReLambda",config.scaling.target_re_lambda)]:self._optional(hit,k,v)
        runtime=SubElement(root,"Runtime")
        for k,v in [("MaxSteps",config.runtime.max_steps),("OutputInterval",config.runtime.output_interval),("DiagnosticsInterval",config.runtime.diagnostics_interval),("CheckpointInterval",config.checkpoint.interval if config.checkpoint.enabled else None),("CheckpointDirectory",config.checkpoint.directory),("CheckpointRetain",config.checkpoint.retain),("SampleStartStep",config.runtime.sample_start_step)]:self._optional(runtime,k,v)
        out=SubElement(root,"Output")
        for k,v in [("Format",config.outputs.format),("WriteVelocity",str(config.outputs.write_velocity).lower()),("WritePressure",str(config.outputs.write_pressure).lower()),("WriteDensity",str(config.outputs.write_density).lower()),("WriteVorticity",str(config.outputs.write_vorticity).lower()),("WriteForcing",str(config.outputs.write_forcing).lower()),("WritePopulations",str(config.outputs.write_populations).lower())]:self._tag(out,k,v)
        indent(root,space="  ");from io import BytesIO;b=BytesIO();ElementTree(root).write(b,encoding="utf-8",xml_declaration=True);return b.getvalue().decode()
    def write_case(self,config,case_dir):
        target=Path(case_dir);target.mkdir(parents=True,exist_ok=True);eff=self.effective_configuration(config)
        paths={"requested":target/"requested_case.json","effective":target/"effective_case.json","xml":target/"case.xml"}
        paths["requested"].write_text(config.model_dump_json(indent=2));paths["effective"].write_text(json.dumps(eff,indent=2));paths["xml"].write_text(self.render_xml(config));return paths
