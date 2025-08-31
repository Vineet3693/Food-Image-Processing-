import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional, List, Dict, Any
import io

# ---------- State ----------
class NutritionState(TypedDict, total=False):
    # Input data
    user_image: Optional[str]
    user_profile: Dict[str, Any]  # age, gender, demography, profession, daily_routine
    medical_report: Optional[str]
    
    # Processing data
    vision_analysis: Dict[str, Any]
    food_identification: Optional[str]
    nutrition_data: Dict[str, Any]
    medical_analysis: Dict[str, Any]
    drug_analysis: List[str]
    health_metrics: Dict[str, Any]
    
    # Validation and safety
    conflicts_detected: List[str]
    risk_assessment: Dict[str, Any]
    safety_status: Literal["safe", "not_safe"]
    
    # Recommendations and analysis
    personalized_recommendations: List[str]
    statistical_analysis: Dict[str, Any]
    
    # Output
    final_report: Dict[str, Any]
    quality_score: float
    output: Dict[str, Any]
    
    # Control flags
    has_clear_image: bool

# ---------- Node Placeholders ----------
def user_image_capture_node(state: NutritionState) -> NutritionState:
    return state

def vision_node(state: NutritionState) -> NutritionState:
    return state

def general_food_name_node(state: NutritionState) -> NutritionState:
    return state

def image_identification_node(state: NutritionState) -> NutritionState:
    return state

def nutrition_analysis_node(state: NutritionState) -> NutritionState:
    return state

def medical_report_processing_node(state: NutritionState) -> NutritionState:
    return state

def medical_report_parsing_node(state: NutritionState) -> NutritionState:
    return state

def drug_analysis_node(state: NutritionState) -> NutritionState:
    return state

def health_metrics_analysis_node(state: NutritionState) -> NutritionState:
    return state

def cross_validation_conflict_resolution_node(state: NutritionState) -> NutritionState:
    return state

def risk_assessment_node(state: NutritionState) -> NutritionState:
    return state

def personalized_recommendation_node(state: NutritionState) -> NutritionState:
    return state

def statistical_analysis_node(state: NutritionState) -> NutritionState:
    return state

def report_generation_node(state: NutritionState) -> NutritionState:
    return state

def quality_assurance_node(state: NutritionState) -> NutritionState:
    return state

def end_unsafe_consumption(state: NutritionState) -> NutritionState:
    return state

# ---------- Routers ----------
def route_after_vision(state: NutritionState) -> str:
    """Route based on image clarity"""
    if state.get("has_clear_image", False):
        return "image_identification"
    else:
        return "general_food_name"

def route_after_risk_assessment(state: NutritionState) -> str:
    """Route based on safety status"""
    safety_status = state.get("safety_status")
    if safety_status == "not_safe":
        return "end_unsafe"
    else:
        return "personalized_recommendations"

# ---------- Build Nutrition Workflow ----------
def build_nutrition_workflow():
    graph = StateGraph(NutritionState)

    # Add all nodes
    graph.add_node("user_image_capture", user_image_capture_node)
    graph.add_node("vision_analysis", vision_node)
    graph.add_node("general_food_name", general_food_name_node)
    graph.add_node("image_identification", image_identification_node)
    graph.add_node("nutrition_analysis", nutrition_analysis_node)
    graph.add_node("medical_report_processing", medical_report_processing_node)
    graph.add_node("medical_report_parsing", medical_report_parsing_node)
    graph.add_node("drug_analysis", drug_analysis_node)
    graph.add_node("health_metrics_analysis", health_metrics_analysis_node)
    graph.add_node("cross_validation_conflict_resolution", cross_validation_conflict_resolution_node)
    graph.add_node("risk_assessment", risk_assessment_node)
    graph.add_node("personalized_recommendations", personalized_recommendation_node)
    graph.add_node("statistical_analysis", statistical_analysis_node)
    graph.add_node("report_generation", report_generation_node)
    graph.add_node("quality_assurance", quality_assurance_node)
    graph.add_node("end_unsafe", end_unsafe_consumption)

    # Sequential flow from start
    graph.add_edge(START, "user_image_capture")
    graph.add_edge("user_image_capture", "vision_analysis")

    # Conditional routing after vision analysis
    graph.add_conditional_edges(
        "vision_analysis",
        route_after_vision,
        {
            "image_identification": "image_identification",
            "general_food_name": "general_food_name"
        }
    )

    # Both paths converge to nutrition analysis
    graph.add_edge("image_identification", "nutrition_analysis")
    graph.add_edge("general_food_name", "nutrition_analysis")

    # Parallel medical processing after nutrition analysis
    graph.add_edge("nutrition_analysis", "medical_report_processing")
    graph.add_edge("medical_report_processing", "medical_report_parsing")
    graph.add_edge("medical_report_parsing", "drug_analysis")
    graph.add_edge("drug_analysis", "health_metrics_analysis")

    # Cross validation and risk assessment
    graph.add_edge("health_metrics_analysis", "cross_validation_conflict_resolution")
    graph.add_edge("cross_validation_conflict_resolution", "risk_assessment")

    # Conditional routing after risk assessment
    graph.add_conditional_edges(
        "risk_assessment",
        route_after_risk_assessment,
        {
            "end_unsafe": "end_unsafe",
            "personalized_recommendations": "personalized_recommendations"
        }
    )

    # Safe path continues through recommendations and analysis
    graph.add_edge("personalized_recommendations", "statistical_analysis")
    graph.add_edge("statistical_analysis", "report_generation")
    graph.add_edge("report_generation", "quality_assurance")

    # End points
    graph.add_edge("quality_assurance", END)
    graph.add_edge("end_unsafe", END)

    return graph.compile()

# ---------- Streamlit App ----------
def main():
    st.title("🍽️ Nutrition Analysis Workflow")
    st.write("Food Image Processing with LangGraph")
    
    # Build workflow
    if st.button("Generate Workflow Diagram"):
        try:
            with st.spinner("Building nutrition workflow..."):
                nutrition_workflow = build_nutrition_workflow()
                
                # Generate flowchart
                mermaid_png = nutrition_workflow.get_graph().draw_mermaid_png()
                
                # Display the image in Streamlit
                st.image(mermaid_png, caption="Nutrition Workflow Diagram", use_column_width=True)
                
                # Provide download button
                st.download_button(
                    label="Download Workflow Diagram",
                    data=mermaid_png,
                    file_name="nutrition_workflow.png",
                    mime="image/png"
                )
                
                st.success("Workflow diagram generated successfully!")
                
        except Exception as e:
            st.error(f"Error generating workflow: {str(e)}")
            st.info("Make sure you have installed all required packages:")
            st.code("pip install langgraph streamlit")

if __name__ == "__main__":
    main()
