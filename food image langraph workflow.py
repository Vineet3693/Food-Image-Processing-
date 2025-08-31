import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Optional, List, Dict, Any

# ---------- State ----------
class FoodImageProcessingState(TypedDict, total=False):
    # Input data
    user_image_unit: Optional[str]
    user_input: Optional[str]
    
    # Validation states
    input_valid: bool
    image_valid: bool
    
    # Processing data
    image_processing_response: Dict[str, Any]
    medical_llm_response: Dict[str, Any]
    
    # Medical data
    medical_report_found: bool
    medical_data_available: bool
    
    # Final outputs
    validated_response: Dict[str, Any]
    personalized_report: Dict[str, Any]
    output: Dict[str, Any]
    
    # Quality metrics
    quality_passed: bool

# ---------- Node Functions ----------
def start_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Initialize the workflow"""
    return state

def user_image_unit_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Process user image input"""
    # Add your image processing logic here
    print("Processing user image unit...")
    return state

def validate_input_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Validate user input"""
    # Add validation logic here
    print("Validating input...")
    # For demo purposes, assume validation passes
    state["input_valid"] = True
    return state

def image_processing_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Process the food image using LLM"""
    print("Processing image with LLM...")
    state["image_processing_response"] = {"status": "processed", "food_items": []}
    return state

def medical_section_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Process medical information"""
    print("Processing medical section...")
    state["medical_llm_response"] = {"status": "analyzed"}
    return state

def personalized_report_generation_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Generate personalized nutrition report"""
    print("Generating personalized report...")
    state["personalized_report"] = {
        "recommendations": [],
        "nutritional_analysis": {},
        "health_insights": []
    }
    return state

def validated_response_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Generate validated response when no medical data"""
    print("Generating validated response...")
    state["validated_response"] = {"status": "validated", "recommendations": []}
    return state

def quality_assurance_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Perform quality assurance checks"""
    print("Performing quality assurance...")
    # Add QA logic here
    state["quality_passed"] = True
    return state

def output_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """Generate final output"""
    print("Generating final output...")
    state["output"] = {
        "final_report": state.get("personalized_report", state.get("validated_response", {})),
        "quality_score": 0.95 if state.get("quality_passed", False) else 0.6
    }
    return state

def end_node(state: FoodImageProcessingState) -> FoodImageProcessingState:
    """End the workflow"""
    print("Workflow completed!")
    return state

# ---------- Router Functions ----------
def route_after_validation(state: FoodImageProcessingState) -> str:
    """Route based on input validation"""
    if state.get("input_valid", False):
        return "valid_path"
    else:
        return "invalid_input"

def route_after_image_processing(state: FoodImageProcessingState) -> str:
    """Route to medical section after image processing"""
    return "medical_section"

def route_medical_report_check(state: FoodImageProcessingState) -> str:
    """Route based on medical report availability"""
    # Simulate medical report check
    medical_report_found = state.get("medical_llm_response", {}).get("medical_data_available", False)
    
    if medical_report_found:
        return "medical_report_found"
    else:
        return "no_medical_data"

def route_quality_check(state: FoodImageProcessingState) -> str:
    """Route based on quality assurance results"""
    if state.get("quality_passed", False):
        return "quality_passed"
    else:
        return "quality_failed"

# ---------- Build Food Image Processing Workflow ----------
def build_food_processing_workflow():
    graph = StateGraph(FoodImageProcessingState)

    # Add all nodes
    graph.add_node("start", start_node)
    graph.add_node("user_image_unit", user_image_unit_node)
    graph.add_node("validate_input", validate_input_node)
    graph.add_node("image_processing", image_processing_node)
    graph.add_node("medical_section", medical_section_node)
    graph.add_node("personalized_report_generation", personalized_report_generation_node)
    graph.add_node("validated_response", validated_response_node)
    graph.add_node("quality_assurance", quality_assurance_node)
    graph.add_node("output", output_node)
    graph.add_node("end", end_node)

    # Define the workflow flow
    graph.add_edge(START, "start")
    graph.add_edge("start", "user_image_unit")
    graph.add_edge("user_image_unit", "validate_input")

    # Conditional routing after validation
    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "valid_path": "image_processing",
            "invalid_input": "end"  # End if invalid input
        }
    )

    # Route to medical section after image processing
    graph.add_edge("image_processing", "medical_section")

    # Conditional routing based on medical report availability
    graph.add_conditional_edges(
        "medical_section",
        route_medical_report_check,
        {
            "medical_report_found": "personalized_report_generation",
            "no_medical_data": "validated_response"
        }
    )

    # Both paths lead to quality assurance
    graph.add_edge("personalized_report_generation", "quality_assurance")
    graph.add_edge("validated_response", "quality_assurance")

    # Quality assurance to output
    graph.add_edge("quality_assurance", "output")
    graph.add_edge("output", "end")
    graph.add_edge("end", END)

    return graph.compile()

# ---------- Streamlit App ----------
def main():
    st.title("🍽️ Food Image Processing LangGraph")
    st.write("AI-powered food analysis and personalized nutrition recommendations")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Food Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
    with col2:
        st.subheader("User Information")
        user_input = st.text_area("Additional information (optional)", 
                                 placeholder="Any dietary restrictions, allergies, or preferences...")

    # Process button
    if st.button("🔍 Analyze Food Image", type="primary"):
        if uploaded_file is not None:
            try:
                with st.spinner("Processing your food image..."):
                    # Build and run workflow
                    workflow = build_food_processing_workflow()
                    
                    # Initial state
                    initial_state = {
                        "user_image_unit": uploaded_file.name,
                        "user_input": user_input
                    }
                    
                    # Run the workflow
                    result = workflow.invoke(initial_state)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Analysis Results")
                        if result.get("output"):
                            st.json(result["output"])
                    
                    with col2:
                        st.subheader("🎯 Quality Score")
                        quality_score = result.get("output", {}).get("quality_score", 0)
                        st.metric("Quality", f"{quality_score:.1%}")
                        
                        if quality_score > 0.8:
                            st.success("High quality analysis!")
                        elif quality_score > 0.6:
                            st.warning("Moderate quality analysis")
                        else:
                            st.error("Low quality analysis - consider uploading a clearer image")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            st.warning("Please upload an image first!")

    # Workflow diagram section
    st.markdown("---")
    if st.button("📊 Generate Workflow Diagram"):
        try:
            with st.spinner("Generating workflow diagram..."):
                workflow = build_food_processing_workflow()
                
                # Generate flowchart
                mermaid_png = workflow.get_graph().draw_mermaid_png()
                
                # Display the image
                st.image(mermaid_png, caption="Food Processing Workflow", use_column_width=True)
                
                # Download button
                st.download_button(
                    label="📥 Download Workflow Diagram",
                    data=mermaid_png,
                    file_name="food_processing_workflow.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"Error generating diagram: {str(e)}")
            st.info("Install required packages: `pip install langgraph streamlit`")

if __name__ == "__main__":
    main()
