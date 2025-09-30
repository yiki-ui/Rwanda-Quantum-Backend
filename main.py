# Rwanda Quantum Agricultural Intelligence Platform
# NISR 2025 Big Data Hackathon - Track 5: Open Innovation
# Quantum molecular simulation for agricultural solutions

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime, date
import json

# Import quantum simulation core
from simulation_core import (
    run_molecule_simulation,
    parse_molecule_string,
    run_bond_scan,
    find_optimized_geometry,
    find_transition_state,
    simplified_molecular_docking,
    predict_material_properties,
    generate_hackathon_dashboard_data
)

app = FastAPI(
    title="Rwanda Quantum Agricultural Intelligence",
    description="Revolutionary agricultural platform using quantum molecular simulation for crop protection, nutrition enhancement, and sustainable farming materials",
    version="2.0.0"
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"  #for test and development do not laugh at me please
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models with Quantum Integration ---

class LocationData(BaseModel):
    latitude: float = Field(..., ge=-2.5, le=-1.0)
    longitude: float = Field(..., ge=28.8, le=30.9)
    altitude: Optional[float] = None
    district: Optional[str] = None
    sector: Optional[str] = None

class MolecularPesticideRequest(BaseModel):
    target_pest: str = Field(..., description="e.g., fall_armyworm, aphids, coffee_berry_borer")
    crop_type: str = Field(..., description="maize, beans, coffee, etc.")
    environmental_safety_level: str = Field("high", description="high, medium, low")
    biodegradability_required: bool = Field(True)
    location: LocationData

class MolecularPesticideResponse(BaseModel):
    success: bool
    recommended_molecule: Optional[str] = None
    molecule_structure: Optional[List[Dict]] = None
    efficacy_prediction: Optional[float] = None
    toxicity_profile: Optional[Dict] = None
    biodegradation_time_days: Optional[float] = None
    binding_affinity_score: Optional[float] = None
    environmental_impact_score: Optional[float] = None
    cost_estimate_usd_per_kg: Optional[float] = None
    synthesis_pathway: Optional[List[str]] = None
    error: Optional[str] = None

class NutrientEnhancementRequest(BaseModel):
    target_crop: str = Field(..., description="crop to enhance")
    deficient_nutrients: List[str] = Field(..., description="List of deficient nutrients")
    enhancement_method: str = Field("biofortification", description="biofortification, foliar_spray, soil_amendment")
    target_increase_percent: float = Field(50, ge=10, le=200)

class NutrientEnhancementResponse(BaseModel):
    success: bool
    enhancement_compounds: Optional[List[Dict]] = None
    molecular_structures: Optional[List[Dict]] = None
    absorption_efficiency: Optional[float] = None
    stability_analysis: Optional[Dict] = None
    application_method: Optional[str] = None
    dosage_recommendations: Optional[Dict] = None
    bioavailability_score: Optional[float] = None
    interaction_warnings: Optional[List[str]] = None
    error: Optional[str] = None

class SustainableMaterialRequest(BaseModel):
    application: str = Field(..., description="packaging, mulch_film, irrigation_tubing, greenhouse_material")
    source_materials: List[str] = Field(..., description="cassava_starch, banana_fiber, coffee_husks")
    required_properties: List[str] = Field(..., description="biodegradable, UV_resistant, water_resistant")
    performance_duration_months: float = Field(6, ge=1, le=36)

class SustainableMaterialResponse(BaseModel):
    success: bool
    material_composition: Optional[str] = None
    molecular_structure: Optional[List[Dict]] = None
    predicted_properties: Optional[Dict] = None
    degradation_pathway: Optional[List[str]] = None
    manufacturing_process: Optional[List[str]] = None
    cost_analysis: Optional[Dict] = None
    environmental_benefits: Optional[List[str]] = None
    performance_metrics: Optional[Dict] = None
    error: Optional[str] = None

class MolecularDockingAnalysisRequest(BaseModel):
    compound_string: str = Field(..., description="Molecular structure of test compound")
    target_site: str = Field(..., description="pest_receptor, nutrient_carrier, plant_membrane")
    analysis_type: str = Field("binding_affinity", description="binding_affinity, toxicity_assessment, absorption_rate")

# --- Quantum Agricultural Functions ---

def design_molecular_pesticide(request: MolecularPesticideRequest) -> Dict[str, Any]:
    """Uses quantum molecular simulation to design targeted, environmentally safe pesticides"""
    try:
        pest_targets = {
            "fall_armyworm": {
                "target_receptor": "sodium_channel",
                "known_inhibitors": ["pyrethroids", "organophosphates"],
                "binding_site": "C 0 0 0; N 1.5 0 0; O 0 1.5 0; S -1.5 0 0"
            },
            "aphids": {
                "target_receptor": "acetylcholine_esterase",
                "known_inhibitors": ["neonicotinoids", "carbamates"],
                "binding_site": "C 0 0 0; N 2.0 0 0; O 0 2.0 0; P -2.0 0 0"
            },
            "coffee_berry_borer": {
                "target_receptor": "chitin_synthesis",
                "known_inhibitors": ["benzoylureas", "chitin_inhibitors"],
                "binding_site": "C 0 0 0; N 1.8 0 0; O 0 1.8 0; F -1.8 0 0"
            }
        }
        
        if request.target_pest.lower() not in pest_targets:
            raise ValueError(f"Target pest {request.target_pest} not in database")
        
        target_info = pest_targets[request.target_pest.lower()]
        
        base_molecules = [
            "C 0 0 0; C 1.5 0 0; N 3.0 0 0; O 1.5 1.5 0; H 0 -1 0; H 1.5 -1 0",
            "C 0 0 0; C 1.4 0 0; C 2.8 0 0; N 4.2 0 0; O 2.8 1.4 0; Cl 0 -1.4 0",
            "C 0 0 0; C 1.3 0.8 0; C 2.6 0 0; N 3.9 0.8 0; O 2.6 1.6 0; F 0 -1.3 0"
        ]
        
        best_molecule = None
        best_score = -float('inf')
        best_binding_affinity = 0
        
        for molecule_string in base_molecules:
            sim_result = run_molecule_simulation(molecule_string, method="hf")
            
            if not sim_result["success"]:
                continue
            
            docking_result = simplified_molecular_docking(
                molecule_string, 
                target_info["binding_site"], 
                num_poses=3
            )
            
            if not docking_result["success"]:
                continue
            
            binding_score = max([pose["score"] for pose in docking_result["poses"]]) if docking_result["poses"] else 0
            
            dipole_magnitude = np.linalg.norm(sim_result["dipole_moment"]) if sim_result["dipole_moment"] else 0
            biodegradability_score = min(100, dipole_magnitude * 20)
            toxicity_score = max(0, 100 - abs(sum([atom.get("charge", 0) for atom in sim_result["atom_data"]])) * 50)
            
            if request.environmental_safety_level == "high":
                total_score = binding_score * 0.4 + biodegradability_score * 0.3 + toxicity_score * 0.3
            else:
                total_score = binding_score * 0.7 + biodegradability_score * 0.2 + toxicity_score * 0.1
            
            if total_score > best_score:
                best_score = total_score
                best_molecule = molecule_string
                best_binding_affinity = binding_score
        
        if not best_molecule:
            raise Exception("No suitable pesticide molecule found")
        
        final_sim = run_molecule_simulation(best_molecule, method="hf")
        
        dipole_mag = np.linalg.norm(final_sim["dipole_moment"]) if final_sim["dipole_moment"] else 0
        biodegradation_time = max(7, 60 - dipole_mag * 15)
        
        num_atoms = len(final_sim["atom_data"])
        complexity_factor = 1 + (num_atoms - 6) * 0.1
        base_cost = 50
        estimated_cost = base_cost * complexity_factor
        
        return {
            "success": True,
            "recommended_molecule": best_molecule,
            "molecule_structure": final_sim["atom_data"],
            "efficacy_prediction": best_binding_affinity,
            "toxicity_profile": {
                "mammalian_toxicity": "low" if toxicity_score > 70 else "moderate",
                "aquatic_toxicity": "low" if biodegradability_score > 50 else "moderate",
                "soil_persistence": "low" if biodegradation_time < 30 else "moderate"
            },
            "biodegradation_time_days": biodegradation_time,
            "binding_affinity_score": best_binding_affinity,
            "environmental_impact_score": (biodegradability_score + toxicity_score) / 2,
            "cost_estimate_usd_per_kg": estimated_cost,
            "synthesis_pathway": [
                "Source renewable feedstock (agricultural waste)",
                "Fermentation-based precursor synthesis", 
                "Green chemistry coupling reactions",
                "Purification and formulation",
                "Quality control and stability testing"
            ]
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def design_nutrient_enhancement(request: NutrientEnhancementRequest) -> Dict[str, Any]:
    """Designs molecular compounds to enhance nutrient content and bioavailability in crops"""
    try:
        nutrient_carriers = {
            "iron": {
                "chelation_agents": [
                    "C 0 0 0; C 1.5 0 0; N 3.0 0 0; O 1.5 1.5 0; Fe 4.5 0.75 0",
                    "C 0 0 0; N 1.4 0 0; N 2.8 0 0; O 1.4 1.4 0; Fe 4.2 0.7 0"
                ],
                "bioavailability_enhancers": ["ascorbic_acid", "citric_acid", "amino_acids"]
            },
            "zinc": {
                "chelation_agents": [
                    "C 0 0 0; C 1.4 0 0; N 2.8 0 0; O 1.4 1.4 0; Zn 4.2 0.7 0",
                    "C 0 0 0; N 1.5 0 0; S 3.0 0 0; O 1.5 1.5 0; Zn 4.5 0.75 0"
                ],
                "bioavailability_enhancers": ["picolinic_acid", "histidine", "cysteine"]
            },
            "vitamin_a": {
                "precursors": [
                    "C 0 0 0; C 1.4 0 0; C 2.8 0 0; C 4.2 0 0; C 5.6 0 0; C 7.0 0 0",
                    "C 0 0 0; C 1.5 0 0; C 3.0 0 0; O 4.5 0 0; C 6.0 0 0; C 7.5 0 0"
                ],
                "stability_enhancers": ["tocopherols", "antioxidants", "encapsulation"]
            }
        }
        
        enhancement_compounds = []
        molecular_structures = []
        
        for nutrient in request.deficient_nutrients:
            if nutrient.lower() in nutrient_carriers:
                carrier_info = nutrient_carriers[nutrient.lower()]
                
                for carrier_molecule in carrier_info.get("chelation_agents", carrier_info.get("precursors", [])):
                    sim_result = run_molecule_simulation(carrier_molecule, method="hf")
                    
                    if not sim_result["success"]:
                        continue
                    
                    dipole_moment = sim_result.get("dipole_moment", [0, 0, 0])
                    polarity = np.linalg.norm(dipole_moment)
                    
                    optimal_polarity = 2.5
                    absorption_efficiency = 100 * np.exp(-((polarity - optimal_polarity)**2) / (2 * 1.0**2))
                    
                    stability_score = 100
                    if sim_result.get("vibrational_frequencies"):
                        min_freq = min(sim_result["vibrational_frequencies"])
                        if min_freq < 200:
                            stability_score = max(50, min_freq / 2)
                    
                    enhancement_compounds.append({
                        "nutrient": nutrient,
                        "compound_type": f"{nutrient}_carrier",
                        "molecule_string": carrier_molecule,
                        "absorption_efficiency": absorption_efficiency,
                        "stability_score": stability_score,
                        "polarity": polarity,
                        "recommended_dosage_ppm": max(1, 100 / absorption_efficiency * 10)
                    })
                    
                    molecular_structures.append({
                        "nutrient": nutrient,
                        "atom_data": sim_result["atom_data"],
                        "energy": sim_result.get("classical_energy", 0),
                        "dipole_moment": dipole_moment
                    })
        
        best_compounds = []
        for nutrient in request.deficient_nutrients:
            nutrient_compounds = [c for c in enhancement_compounds if c["nutrient"] == nutrient]
            if nutrient_compounds:
                best_compound = max(nutrient_compounds, key=lambda x: x["absorption_efficiency"] * x["stability_score"])
                best_compounds.append(best_compound)
        
        if request.enhancement_method == "biofortification":
            application_method = "Genetic enhancement of biosynthesis pathways"
        elif request.enhancement_method == "foliar_spray":
            application_method = "Foliar application during vegetative growth"
        else:
            application_method = "Soil incorporation before planting"
        
        avg_absorption = np.mean([c["absorption_efficiency"] for c in best_compounds]) if best_compounds else 0
        avg_stability = np.mean([c["stability_score"] for c in best_compounds]) if best_compounds else 0
        bioavailability_score = (avg_absorption + avg_stability) / 2
        
        return {
            "success": True,
            "enhancement_compounds": best_compounds,
            "molecular_structures": molecular_structures[:10],
            "absorption_efficiency": avg_absorption,
            "stability_analysis": {
                "thermal_stability": "high" if avg_stability > 80 else "moderate",
                "oxidation_resistance": "high" if avg_stability > 70 else "moderate",
                "ph_stability": "moderate"
            },
            "application_method": application_method,
            "dosage_recommendations": {
                nutrient: f"{compound['recommended_dosage_ppm']:.1f} ppm" 
                for compound in best_compounds for nutrient in [compound["nutrient"]]
            },
            "bioavailability_score": bioavailability_score,
            "interaction_warnings": [
                "Monitor soil pH after application",
                "Avoid mixing with high-phosphate fertilizers",
                "Apply during optimal weather conditions"
            ]
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def design_sustainable_agricultural_material(request: SustainableMaterialRequest) -> Dict[str, Any]:
    """Uses existing predict_material_properties function to design sustainable agricultural materials"""
    try:
        source_molecules = {
            "cassava_starch": "C 6 0 0; O 0 3 0; H 3 3 0; H 9 3 0; H 6 6 0; H 6 -3 0",
            "banana_fiber": "C 0 0 0; C 1.4 0 0; O 2.8 0 0; H 0 1.4 0; H 1.4 1.4 0; H 2.8 1.4 0",
            "coffee_husks": "C 0 0 0; C 1.5 0.8 0; O 3.0 0 0; N 1.5 -1.2 0; H 0 1.4 0; H 3.0 1.4 0"
        }
        
        selected_molecules = []
        for source in request.source_materials:
            if source.lower() in source_molecules:
                selected_molecules.append(source_molecules[source.lower()])
        
        if not selected_molecules:
            raise ValueError("No valid source materials provided")
        
        base_molecule = selected_molecules[0]
        
        material_analysis = predict_material_properties(
            base_molecule, 
            num_repeats=min(4, len(selected_molecules) + 1)
        )
        
        if not material_analysis["success"]:
            raise Exception(f"Material analysis failed: {material_analysis.get('error', 'Unknown error')}")
        
        predicted_properties = {}
        
        energy_density = material_analysis.get("conceptual_energy_density", 0)
        dipole_magnitude = material_analysis.get("conceptual_avg_dipole_moment_magnitude", 0)
        
        biodegradation_months = max(1, 12 * np.exp(-dipole_magnitude / 2))
        predicted_properties["biodegradation_time_months"] = biodegradation_months
        predicted_properties["biodegradable"] = biodegradation_months <= request.performance_duration_months * 2
        
        max_freq = material_analysis.get("conceptual_highest_vibrational_frequency", 0)
        uv_resistance_score = min(100, max_freq / 40)
        predicted_properties["uv_resistance_score"] = uv_resistance_score
        predicted_properties["UV_resistant"] = uv_resistance_score > 60
        
        water_resistance_score = max(0, 100 - dipole_magnitude * 15)
        predicted_properties["water_resistance_score"] = water_resistance_score
        predicted_properties["water_resistant"] = water_resistance_score > 50
        
        manufacturing_steps = [
            "Collection and preparation of agricultural waste",
            "Biomass preprocessing and size reduction",
            "Chemical treatment for fiber extraction",
            "Molecular cross-linking and polymerization",
            "Forming and shaping processes",
            "Quality control and testing",
            "Packaging for distribution"
        ]
        
        raw_material_cost = len(selected_molecules) * 0.5
        processing_cost = 2.0 + (len(manufacturing_steps) * 0.3)
        total_cost = raw_material_cost + processing_cost
        
        cost_analysis = {
            "raw_materials_usd_per_kg": raw_material_cost,
            "processing_cost_usd_per_kg": processing_cost,
            "total_cost_usd_per_kg": total_cost,
            "cost_comparison_conventional": total_cost / 8.0
        }
        
        environmental_benefits = [
            f"Reduces agricultural waste by utilizing {', '.join(request.source_materials)}",
            f"Biodegrades in {biodegradation_months:.1f} months vs. centuries for plastics",
            "Zero petroleum-based inputs required",
            "Carbon neutral or negative lifecycle",
            "Supports circular economy in agriculture",
            "Creates additional income stream for farmers"
        ]
        
        performance_metrics = {
            "strength_mpa": max(10, energy_density * 2),
            "flexibility_score": min(100, dipole_magnitude * 25),
            "durability_months": min(request.performance_duration_months * 1.2, biodegradation_months * 0.8),
            "temperature_resistance_celsius": 60 + max_freq / 100
        }
        
        return {
            "success": True,
            "material_composition": f"Bio-composite from {', '.join(request.source_materials)}",
            "molecular_structure": material_analysis.get("cluster_atom_data", [])[:50],
            "predicted_properties": predicted_properties,
            "degradation_pathway": [
                "Hydrolysis of polymer chains",
                "Microbial breakdown of oligomers",
                "Mineralization to CO2 and H2O",
                "Complete biodegradation"
            ],
            "manufacturing_process": manufacturing_steps,
            "cost_analysis": cost_analysis,
            "environmental_benefits": environmental_benefits,
            "performance_metrics": performance_metrics
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# --- API Endpoints ---

@app.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "Rwanda Quantum Agricultural Intelligence Platform",
        "version": "2.0.0",
        "status": "operational",
        "hackathon": "NISR 2025 Big Data Hackathon"
    }

@app.get("/health", summary="Health Check")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/generate_hackathon_dashboard_data", summary="Generate Hackathon Dashboard Data")
async def get_dashboard_data_endpoint():
    """Retrieves comprehensive data for the frontend hackathon dashboard"""
    try:
        data = generate_hackathon_dashboard_data()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design_molecular_pesticide", response_model=MolecularPesticideResponse, summary="Design Quantum-Optimized Pesticides")
async def design_molecular_pesticide_endpoint(request: MolecularPesticideRequest):
    """Revolutionary approach: Uses quantum molecular simulation to design targeted, biodegradable pesticides"""
    try:
        result = design_molecular_pesticide(request)
        return MolecularPesticideResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design_nutrient_enhancement", response_model=NutrientEnhancementResponse, summary="Molecular Nutrition Enhancement")
async def design_nutrient_enhancement_endpoint(request: NutrientEnhancementRequest):
    """Addresses Track 2 (Hidden Hunger): Uses molecular simulation to design compounds that enhance nutrient bioavailability"""
    try:
        result = design_nutrient_enhancement(request)
        return NutrientEnhancementResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design_sustainable_material", response_model=SustainableMaterialResponse, summary="Quantum-Designed Agricultural Materials")
async def design_sustainable_material_endpoint(request: SustainableMaterialRequest):
    """Revolutionary material design: Uses quantum material properties prediction to design sustainable agricultural materials"""
    try:
        result = design_sustainable_agricultural_material(request)
        return SustainableMaterialResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate", summary="Core Quantum Molecular Simulation")
async def run_simulation_endpoint(request: dict):
    """Your original quantum/classical molecular simulation endpoint"""
    try:
        from pydantic import BaseModel
        class SimRequest(BaseModel):
            molecule_string: str = "H 0 0 0; H 0 0 0.74"
            method: str = "vqe"
            bond_distance_scale: float = 1.0
            
        sim_request = SimRequest(**request)
        results = run_molecule_simulation(
            sim_request.molecule_string,
            sim_request.method,
            sim_request.bond_distance_scale
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/molecular_docking", summary="Molecular Docking Analysis")
async def molecular_docking_endpoint(request: MolecularDockingAnalysisRequest):
    """Your original molecular docking, now applied to agricultural compounds"""
    try:
        if request.target_site == "pest_receptor":
            protein_site = "C 0 0 0; N 1.5 0 0; O 0 1.5 0; S -1.5 0 0"
        elif request.target_site == "nutrient_carrier":
            protein_site = "C 0 0 0; N 2.0 0 0; O 0 2.0 0; P -2.0 0 0"
        else:
            protein_site = "C 0 0 0; C 1.4 0 0; O 2.8 0 0; N 1.4 1.4 0"
            
        results = simplified_molecular_docking(
            request.compound_string,
            protein_site,
            num_poses=5
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
