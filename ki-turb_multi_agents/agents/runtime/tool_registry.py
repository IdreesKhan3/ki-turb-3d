"""Declarative tool registry and strict per-agent permissions."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any,Callable,Dict,FrozenSet,List,Optional
from .models import AgentName
_WEB_RESEARCH_TOOLS={"web_search","search_research_papers","browse_web"}
_ANALYST_TOOLS={"list_directory","find_file","read_file","read_document","list_simulation_jobs","load_analysis_products","get_analysis_product_summary","compute_overview_validation","compute_pdfs","compute_volume_field","compute_spectra","compute_spectral_isotropy","compute_isotropy","compute_flatness","compute_structure_functions","export_data","web_search","search_research_papers","browse_web","download_file","semantic_search","find_symbol_definitions","find_symbol_references","write_file","generate_content","generate_code","compile_latex"}
_VISUALIZER_TOOLS={"plot_spectrum","get_energy_spectra_theory","plot_spectral_isotropy","plot_component_spectra","get_spectral_isotropy_summary","get_spectral_isotropy_theory","plot_real_isotropy","plot_lumley_triangle","plot_diagonal_bii","plot_cross_correlations","plot_deviations","plot_convergence","get_real_isotropy_summary","get_real_isotropy_theory","export_figure","export_data","export_isotropy_data","get_overview_summary","get_overview_theory","get_analysis_product_summary","get_theory_ns_equations","get_theory_lbm_formulation","plot_d3q19_lattice","get_theory_mrt_matrix","plot_flatness","get_flatness_summary","get_flatness_theory","export_flatness_data","plot_structure_functions","get_structure_functions_theory","plot_turbulence_stats","get_turbulence_stats_columns","get_turbulence_stats_summary","plot_volume_3d","get_volume_viewer_theory","plot_pdf","add_report_section","generate_report","preview_report","remove_report_section","reorder_report_section","edit_report_section"}
_STEWARD_TOOLS={"list_directory","find_file","read_file","read_document","set_app_theme","load_data","load_dataset_manifest","list_simulation_jobs","load_analysis_products","get_analysis_product_summary","set_selection_mode","set_hdf5_format","search_codebase","extract_section","regex_search","git_operation","delete_file","modify_file","rename_file","write_file","run_shell_command","web_search","search_research_papers","browse_web","download_file","run_pytest","run_import_check","run_verify_command"}
_ENGINEER_TOOLS={"list_directory","find_file","read_file","search_codebase","extract_section","regex_search","semantic_search","find_symbol_definitions","find_symbol_references","git_operation","delete_file","modify_file","rename_file","write_file","run_shell_command","web_search","search_research_papers","browse_web","download_file","run_pytest","run_import_check","run_verify_command"}
_SIMULATION_TOOLS={"build_simulation_case","compile_simulation","start_simulation","check_simulation_status","cancel_simulation","supervise_simulation","fetch_simulation_outputs","postprocess_simulation_outputs","read_dataset_manifest","load_dataset_manifest","list_simulation_jobs","load_analysis_products","get_analysis_product_summary","web_search","browse_web","search_research_papers"}
_ORCHESTRATOR_TOOLS=set(_WEB_RESEARCH_TOOLS)
_REVIEWER_TOOLS=set(_WEB_RESEARCH_TOOLS)
# Intentionally NOT confirmable (allowlist/sandbox is the safety net):
# - run_shell_command: read-only inspection; confirming blocked cmds caused Accept loops
# - run_verify_command / run_pytest / run_import_check: scoped self-checks after edits;
#   confirming them after every write_file caused overlapping Accept spam with Retrieve
_CONFIRMABLE_TOOLS={"delete_file","rename_file","write_file","modify_file","download_file","build_simulation_case","compile_simulation","start_simulation","cancel_simulation","fetch_simulation_outputs","postprocess_simulation_outputs"}
_AGENT_GROUPS={AgentName.STEWARD:_STEWARD_TOOLS,AgentName.ANALYST:_ANALYST_TOOLS,AgentName.VISUALIZER:_VISUALIZER_TOOLS,AgentName.SIMULATION:_SIMULATION_TOOLS,AgentName.ORCHESTRATOR:_ORCHESTRATOR_TOOLS,AgentName.REVIEWER:_REVIEWER_TOOLS,AgentName.ENGINEER:_ENGINEER_TOOLS}
@dataclass
class ToolSpec:
    name:str; allowed_agents:FrozenSet[AgentName]=frozenset(); confirmation_required:bool=False
    description:str=""; schema:Optional[Dict[str,Any]]=None; executor:Optional[Callable[...,Any]]=None
    tags:FrozenSet[str]=field(default_factory=frozenset)
    def allows(self,agent:Any)->bool:
        role=AgentName.coerce(agent); return role is not None and role in self.allowed_agents
def _build_registry():
    names=set().union(*_AGENT_GROUPS.values(),_CONFIRMABLE_TOOLS)
    return {name:ToolSpec(name=name,allowed_agents=frozenset(a for a,g in _AGENT_GROUPS.items() if name in g),confirmation_required=name in _CONFIRMABLE_TOOLS) for name in sorted(names)}
TOOL_REGISTRY=_build_registry()
def get_spec(name):return TOOL_REGISTRY.get(name)
def all_registered_tools():return frozenset(TOOL_REGISTRY)
def tools_for_agent(agent):
    role=AgentName.coerce(agent)
    return frozenset(name for name,spec in TOOL_REGISTRY.items() if role is not None and role in spec.allowed_agents)
def is_agent_allowed(tool,agent):
    spec=get_spec(tool); return bool(spec and spec.allows(agent))
def requires_confirmation(tool):
    spec=get_spec(tool); return bool(spec and spec.confirmation_required)
def confirmable_tools():return frozenset(name for name,s in TOOL_REGISTRY.items() if s.confirmation_required)
def register(spec,*,overwrite=False):
    if spec.name in TOOL_REGISTRY and not overwrite:raise ValueError(f"Tool '{spec.name}' is already registered")
    TOOL_REGISTRY[spec.name]=spec; return spec
def enrich_from_definitions(definitions:List[Dict[str,Any]]):
    for d in definitions or []:
        spec=TOOL_REGISTRY.get(d.get("name"))
        if spec:
            if not spec.description:spec.description=d.get("description","")
            if spec.schema is None:spec.schema=d.get("parameters")
