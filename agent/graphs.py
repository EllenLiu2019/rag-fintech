from contextlib import asynccontextmanager
from agent.medical_agent import MedicalAgent


@asynccontextmanager
async def medical_agent():
    med = MedicalAgent()
    try:
        yield med.agent
    finally:
        pass
