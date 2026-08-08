import sys
import io
import traceback

class CodeExecutor:
    @staticmethod
    def execute_python_code(code_str: str) -> tuple[str, bool]:
        clean_code = code_str.strip()
        
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        
        try:
            exec_globals = {}
            exec(clean_code, exec_globals)
            
            sys.stdout = old_stdout
            output = buffer.getvalue().strip()
            
            return output if output else "Code executed successfully with no print output.", True
            
        except Exception:
            sys.stdout = old_stdout
            error_msg = traceback.format_exc()
            return error_msg.strip(), False
        
        
        
# import requests

# class CodeExecutor:
#     @staticmethod
#     def execute_python_code(code_str: str) -> tuple[str, bool]:
#         clean_code = code_str.replace("```python", "").replace("```", "").strip()
        
        
#         url = "https://ce.judge0.com/submissions?base64_encoded=false&wait=true"
        
#         headers = {
#             "Content-Type": "application/json"
#         }

#         try:
#             response = requests.post(
#                 url, 
#                 json={"language_id": 71, "source_code": clean_code}, 
#                 headers=headers, 
#                 timeout=15
#             )
#             response.raise_for_status()
#             result = response.json()
            
#             if result.get("status", {}).get("id") != 3:
#                 error = result.get("stderr") or result.get("compile_output") or result.get("message") or "Execution failed."
#                 return error.strip(), False
                
#             stdout = result.get("stdout") or ""
#             return stdout.strip() if stdout.strip() else "Code executed successfully with no output.", True

#         except requests.exceptions.RequestException as e:
#             return f"Network or API Error: {str(e)}", False
#         except Exception as e:
#             return f"Execution Error: {str(e)}", False