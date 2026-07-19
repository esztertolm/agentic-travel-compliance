import sys
import os
import time
import statistics
import json
import uuid
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add both the root and the src directory to sys.path
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.agent import AGENT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("LoadTest")

def run_load_test(num_iterations: int = 4):
    logger.info(f"=== Starting Load & Functional Test with {num_iterations} iterations ===")
    
    test_queries = [
        # --- RAG Only Queries ---
        "What is the maximum allowance for a hotel stay per night?",
        "Can I get reimbursed for laundry services during my business trip?",
        "How do I book a rental car according to the company policy?",
        "Is business class allowed for domestic flights?",
        "What happens if I combine personal travel with business travel?",
        "Are in-room movies or spa charges reimbursable?",
        
        # --- Currency Tool Only Queries ---
        "Convert 250 EUR to USD using the latest exchange rate.",
        "How much was 500 EUR in USD on 2023-05-10?",
        "I travelled on 2022-11-15. How much was 1000 GBP in EUR back then?",
        "Convert 150000 HUF to USD on 2024-01-05.",
        
        # --- Multi-Tool (RAG + Currency) Edge Cases ---
        "I spent 18000 HUF on dinner yesterday. Is this fully reimbursable under the meal allowance?",
        "My breakfast in London cost 35 GBP on 2023-10-05. Does the travel policy cover this amount?",
        "I paid 150 PLN for a client lunch. What is the policy on this, and what is the equivalent in EUR?"
    ]
    
    results = []
    latencies = []
    
    # 1. Define file path at the beginning so we can incrementally write to it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(os.path.dirname(__file__), f"test_results_{timestamp}.json")
    
    for i in range(num_iterations):
        query = test_queries[i % len(test_queries)]
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        state = {"messages": [HumanMessage(content=query)]}
        
        logger.info(f"Iteration {i+1}/{num_iterations} - Testing query: '{query}'")
        
        start_time = time.time()
        
        try:
            response = AGENT.invoke(state, config=config)
            end_time = time.time()
            
            duration = end_time - start_time
            latencies.append(duration)
            
            final_answer = response["messages"][-1].content
            status = "Success"
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            latencies.append(duration)
            final_answer = str(e)
            status = "Failed"
            logger.error(f"Iteration {i+1} failed: {e}")

        # Append current iteration data
        results.append({
            "iteration": i + 1,
            "query": query,
            "latency_seconds": round(duration, 2),
            "status": status,
            "full_response": final_answer 
        })
        
        # Calculate intermediate statistics for the live backup
        avg_latency_so_far = statistics.mean(latencies)
        
        intermediate_report = {
            "meta": {
                "status": "In Progress",
                "last_updated": datetime.now().isoformat(),
                "progress": f"{i+1}/{num_iterations}"
            },
            "statistics": {
                "total_requests_so_far": i + 1,
                "successful_requests": sum(1 for r in results if r["status"] == "Success"),
                "failed_requests": sum(1 for r in results if r["status"] == "Failed"),
                "average_latency_sec": round(avg_latency_so_far, 2)
            },
            "details": results
        }
        
        # Incremental write: updates the file with latest state after EVERY request
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(intermediate_report, f, indent=4)
        
        time.sleep(1)

    # 2. Final calculations and structured report once all iterations are done
    min_latency = min(latencies)
    max_latency = max(latencies)
    p95_latency = statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else max_latency

    final_report = {
        "meta": {
            "status": "Completed",
            "completed_at": datetime.now().isoformat(),
            "progress": f"{num_iterations}/{num_iterations}"
        },
        "statistics": {
            "total_requests": num_iterations,
            "successful_requests": sum(1 for r in results if r["status"] == "Success"),
            "failed_requests": sum(1 for r in results if r["status"] == "Failed"),
            "average_latency_sec": round(statistics.mean(latencies), 2),
            "min_latency_sec": round(min_latency, 2),
            "max_latency_sec": round(max_latency, 2),
            "p95_latency_sec": round(p95_latency, 2)
        },
        "details": results
    }

    # Final safe overwrite with full metrics
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
        
    logger.info("=== Load Test Complete ===")
    logger.info(f"Detailed results safely saved to {output_filename}")

if __name__ == "__main__":
    run_load_test(num_iterations=50)