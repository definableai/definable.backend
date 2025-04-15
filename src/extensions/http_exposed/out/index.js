"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.deactivate = exports.activate = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
// Define decoration types
let getEndpointDecoration;
let postEndpointDecoration;
let putEndpointDecoration;
let deleteEndpointDecoration;
let activeEditor;
let decorationsTimeout;
function activate(context) {
    console.log('Endpoint Navigator extension activated');
    // Create decoration for different HTTP methods with cleaner styling
    getEndpointDecoration = vscode.window.createTextEditorDecorationType({
        color: '#4EC9B0',
        fontStyle: 'italic',
        textDecoration: 'none',
        backgroundColor: 'rgba(78, 201, 176, 0.05)' // Very light teal background
    });
    postEndpointDecoration = vscode.window.createTextEditorDecorationType({
        color: '#569CD6',
        fontStyle: 'italic',
        textDecoration: 'none',
        backgroundColor: 'rgba(86, 156, 214, 0.05)' // Very light blue background
    });
    putEndpointDecoration = vscode.window.createTextEditorDecorationType({
        color: '#C586C0',
        fontStyle: 'italic',
        textDecoration: 'none',
        backgroundColor: 'rgba(197, 134, 192, 0.05)' // Very light purple background
    });
    deleteEndpointDecoration = vscode.window.createTextEditorDecorationType({
        color: '#F14C4C',
        fontStyle: 'italic',
        textDecoration: 'none',
        backgroundColor: 'rgba(241, 76, 76, 0.05)' // Very light red background
    });
    // Update decorations when the active editor changes
    vscode.window.onDidChangeActiveTextEditor(editor => {
        activeEditor = editor;
        if (editor) {
            updateDecorations();
        }
    }, null, context.subscriptions);
    // Update decorations when the document changes
    vscode.workspace.onDidChangeTextDocument(event => {
        if (activeEditor && event.document === activeEditor.document) {
            updateDecorations();
        }
    }, null, context.subscriptions);
    // Set initial active editor
    activeEditor = vscode.window.activeTextEditor;
    if (activeEditor) {
        updateDecorations();
    }
    // Register the definition provider for Python files
    const provider = vscode.languages.registerDefinitionProvider('python', {
        async provideDefinition(document, position, token) {
            console.log(`Definition requested at line ${position.line + 1}, character ${position.character}`);
            // Get current line text
            const line = document.lineAt(position.line).text;
            console.log(`Line content: ${line}`);
            // Check if the line contains a string with the pattern "method=name"
            if (!line.includes('=')) {
                console.log('Line does not contain "=" character, skipping');
                return null;
            }
            // Find all strings in the line
            const stringMatches = [...line.matchAll(/"([^"]+)"/g)];
            const cursorOffset = document.offsetAt(position) - document.offsetAt(position.with(undefined, 0));
            // Check if cursor is inside any of the strings
            let endpoint = null;
            for (const match of stringMatches) {
                const startPos = match.index || 0;
                const endPos = startPos + match[0].length;
                // If cursor is inside this string
                if (cursorOffset >= startPos && cursorOffset <= endPos) {
                    console.log(`Cursor is inside string: "${match[1]}"`);
                    // Check if it's an endpoint pattern
                    const patternMatch = match[1].match(/(get|post|put|delete)=([a-z_]+)/);
                    if (patternMatch) {
                        endpoint = {
                            httpMethod: patternMatch[1],
                            methodName: patternMatch[2]
                        };
                        break;
                    }
                }
            }
            if (!endpoint) {
                console.log('No valid endpoint found at cursor position');
                return null;
            }
            const targetMethodName = `${endpoint.httpMethod}_${endpoint.methodName}`;
            console.log(`Looking for method: ${targetMethodName}`);
            // Search for the method in the current file
            const fileContent = document.getText();
            const methodPattern = new RegExp(`\\basync\\s+def\\s+${targetMethodName}\\s*\\(`, 'm');
            const methodMatch = methodPattern.exec(fileContent);
            if (methodMatch) {
                console.log(`Method found in current file at index ${methodMatch.index}`);
                const methodPos = document.positionAt(methodMatch.index);
                return new vscode.Location(document.uri, methodPos);
            }
            console.log('Method not found in current file, searching workspace...');
            // If not found in current file, search other Python files
            try {
                const files = await vscode.workspace.findFiles('**/services/**/*.py', '**/node_modules/**');
                for (const file of files) {
                    try {
                        // Skip the current file (already checked)
                        if (file.fsPath === document.uri.fsPath) {
                            continue;
                        }
                        const content = fs.readFileSync(file.fsPath, 'utf8');
                        // Simple check if method name appears in file
                        if (content.includes(`def ${targetMethodName}(`)) {
                            // Determine the exact line number
                            const lines = content.split('\n');
                            for (let i = 0; i < lines.length; i++) {
                                if (lines[i].includes(`def ${targetMethodName}(`)) {
                                    console.log(`Method found in ${file.fsPath} at line ${i + 1}`);
                                    return new vscode.Location(vscode.Uri.file(file.fsPath), new vscode.Position(i, 0));
                                }
                            }
                        }
                    }
                    catch (err) {
                        console.error(`Error reading file ${file.fsPath}: ${err}`);
                    }
                }
                console.log('Method not found in workspace');
            }
            catch (err) {
                console.error(`Error searching files: ${err}`);
            }
            return null;
        }
    });
    context.subscriptions.push(provider);
    console.log('Endpoint Navigator extension is now active');
}
exports.activate = activate;
/**
 * Update decorations in the active editor
 */
function updateDecorations() {
    if (!activeEditor || activeEditor.document.languageId !== 'python') {
        return;
    }
    // Clear the timeout if it exists
    if (decorationsTimeout) {
        clearTimeout(decorationsTimeout);
        decorationsTimeout = undefined;
    }
    // Set a timeout to update decorations
    decorationsTimeout = setTimeout(() => {
        try {
            console.log('Updating decorations');
            // Create separate arrays for different HTTP methods
            const getDecorations = [];
            const postDecorations = [];
            const putDecorations = [];
            const deleteDecorations = [];
            const document = activeEditor.document;
            const text = document.getText();
            // First, find all http_exposed lists in the file
            const lines = text.split('\n');
            const httpExposedRanges = [];
            let inHttpExposed = false;
            let startLine = -1;
            let bracketCount = 0;
            // Scan the document to find http_exposed lists and their ranges
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                // Start of http_exposed list
                if (line.includes('http_exposed') && line.includes('=') && line.includes('[')) {
                    inHttpExposed = true;
                    startLine = i;
                    bracketCount = (line.match(/\[/g) || []).length - (line.match(/\]/g) || []).length;
                    // If the list is on a single line
                    if (bracketCount === 0 && line.includes(']')) {
                        httpExposedRanges.push({ start: startLine, end: i });
                        inHttpExposed = false;
                    }
                    continue;
                }
                // Count brackets to track nested lists
                if (inHttpExposed) {
                    bracketCount += (line.match(/\[/g) || []).length - (line.match(/\]/g) || []).length;
                    // End of http_exposed list
                    if (bracketCount <= 0 && line.includes(']')) {
                        httpExposedRanges.push({ start: startLine, end: i });
                        inHttpExposed = false;
                    }
                }
            }
            console.log(`Found ${httpExposedRanges.length} http_exposed ranges`);
            // Find all endpoint strings in the document
            // This improved regex matches endpoints with single or double quotes
            const endpointRegex = /["'](get|post|put|delete)=([a-z_]+)["']/g;
            let match;
            while ((match = endpointRegex.exec(text)) !== null) {
                // Get the position of the match
                const matchStart = match.index;
                const matchEnd = matchStart + match[0].length;
                const startPos = document.positionAt(matchStart);
                const endPos = document.positionAt(matchEnd);
                // Check if this endpoint is within any http_exposed range
                const lineNumber = startPos.line;
                let isInHttpExposed = false;
                for (const range of httpExposedRanges) {
                    if (lineNumber >= range.start && lineNumber <= range.end) {
                        isInHttpExposed = true;
                        break;
                    }
                }
                // Fallback: check surrounding lines if we didn't find it in an http_exposed range
                if (!isInHttpExposed) {
                    // Check current line
                    const lineText = document.lineAt(lineNumber).text;
                    if (lineText.includes('http_exposed')) {
                        isInHttpExposed = true;
                    }
                    else {
                        // Check a few lines before for http_exposed
                        for (let i = 1; i <= 15; i++) {
                            if (lineNumber - i < 0)
                                break;
                            const prevLine = document.lineAt(lineNumber - i).text;
                            if (prevLine.includes('http_exposed')) {
                                isInHttpExposed = true;
                                break;
                            }
                        }
                    }
                }
                // Extra debug for problematic lines (56-61)
                if (lineNumber >= 55 && lineNumber <= 62) {
                    console.log(`Line ${lineNumber + 1}: "${lines[lineNumber].trim()}" - isInHttpExposed: ${isInHttpExposed}`);
                }
                // Only add decoration if it's in an http_exposed context
                if (isInHttpExposed) {
                    // Extract the method and endpoint name - group 1 is the method, group 2 is the endpoint name
                    const httpMethod = match[1].toLowerCase(); // get, post, put, delete
                    const endpointName = match[2]; // the name part after the =
                    // Create a decoration with improved hover message
                    const decoration = {
                        range: new vscode.Range(startPos, endPos),
                        hoverMessage: new vscode.MarkdownString(`**${httpMethod.toUpperCase()} Endpoint**: \`${endpointName}\`\n\n` +
                            `**Method**: \`${httpMethod}_${endpointName}\`\n\n` +
                            `*Ctrl+Click to navigate to implementation*`)
                    };
                    // Add to the appropriate decoration array based on HTTP method
                    switch (httpMethod) {
                        case 'get':
                            getDecorations.push(decoration);
                            break;
                        case 'post':
                            postDecorations.push(decoration);
                            break;
                        case 'put':
                            putDecorations.push(decoration);
                            break;
                        case 'delete':
                            deleteDecorations.push(decoration);
                            break;
                    }
                }
            }
            console.log(`Found ${getDecorations.length} GET, ${postDecorations.length} POST, ${putDecorations.length} PUT, ${deleteDecorations.length} DELETE endpoints`);
            // Apply decorations for each HTTP method
            activeEditor.setDecorations(getEndpointDecoration, getDecorations);
            activeEditor.setDecorations(postEndpointDecoration, postDecorations);
            activeEditor.setDecorations(putEndpointDecoration, putDecorations);
            activeEditor.setDecorations(deleteEndpointDecoration, deleteDecorations);
        }
        catch (e) {
            console.error('Error updating decorations:', e);
        }
    }, 100);
}
function deactivate() {
    console.log('Endpoint Navigator extension deactivated');
    // Dispose of decorations
    if (getEndpointDecoration)
        getEndpointDecoration.dispose();
    if (postEndpointDecoration)
        postEndpointDecoration.dispose();
    if (putEndpointDecoration)
        putEndpointDecoration.dispose();
    if (deleteEndpointDecoration)
        deleteEndpointDecoration.dispose();
    // Clear timeout
    if (decorationsTimeout) {
        clearTimeout(decorationsTimeout);
        decorationsTimeout = undefined;
    }
}
exports.deactivate = deactivate;
//# sourceMappingURL=index.js.map