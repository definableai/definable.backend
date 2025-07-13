import express from "express";
import type { Request, Response } from "express";
import bodyParser from "body-parser";
import { CodeSandbox } from "@codesandbox/sdk";

const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Endpoint to run Python code
// todo : add proper response types
app.post("/run-python", async (req: Request, res: Response): Promise<any>  => {
  const { requirements, tool_code, agent_code, version, tool_name } = req.body;

  if (!requirements || !tool_code || !agent_code) {
    return res.status(400).json({ error: "Both 'requirements', 'tool_code', and 'agent_code' are required." });
  }

  try {
    // Create the client with your token
    const sdk = new CodeSandbox("csb_v1_kYUyJZYxotssK7cgHHQGbBeNh4ymxj-FCjNdP7qa5S8");

    // Open the sandbox
    const sandbox = await sdk.sandbox.open("3kx8gq");
    console.log("Sandbox ID:", sandbox.id);
    
    const formattedRequirements = Array.isArray(requirements)
      ? requirements.map(req => `"${req}"`).join(" ")
      : requirements;
    // Install Python dependencies
    // todo : isolated environments
    await sandbox.shells.run(`pip install ${formattedRequirements}`);
    await sandbox.shells.run(`mkdir -p ${tool_name}/${version}`);
    await sandbox.fs.writeTextFile(`./${tool_name}/${version}/tool.py`, tool_code);
    await sandbox.fs.writeTextFile(`./${tool_name}/${version}/agent.py`, agent_code);
    // Run the Python code

    const output = await sandbox.shells.run(`python ./${tool_name}/${version}/agent.py`);

    // Disconnect from the sandbox
    await sandbox.disconnect();

    // Return the output
    res.json(output);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "An error occurred while running the Python code." });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
