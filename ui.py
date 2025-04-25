import chainlit as ch
import aiohttp
import datetime
import asyncio
import json


url = "http://127.0.0.1:5000/api/query"  # Using 127.0.0.1 instead of localhost

@ch.on_chat_start
async def start_chat():
    time = datetime.datetime.now().hour
    if time < 12:
        greeting = "Good Morning! How can I assist you today?"
    elif time >= 12 and time < 18:
        greeting = "Good Afternoon! How can I assist you today?"
    else: 
        greeting = "Good Evening! How can I assist you today?"
    await ch.Message(content=greeting).send()


@ch.on_message
async def main(message):
    # Send a processing message
    await ch.Message(content="Processing your query... This may take several minutes as the model is running locally.").send()
    
    try:
        # Use aiohttp for async HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json={"question": message.content}, 
                timeout=aiohttp.ClientTimeout(total=600)  # 10-minute timeout
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Send success message
                    await ch.Message(content="Query processed successfully!").send()
                    
                    # Extract SQL query and results from response
                    sql_query = data.get('sql', '')
                    table_results = data.get('results', [])
                    
                    # Format the response with markdown for SQL query
                    formatted_response = f"""
### SQL Query
```sql
{sql_query}
```
"""
                    # Send the formatted text response with SQL query
                    await ch.Message(content=formatted_response).send()
                    
                    # Format table results as markdown text table instead of using Table component
                    if table_results and len(table_results) > 0:
                        # Get headers from first result
                        headers = list(table_results[0].keys())
                        
                        # Create markdown table header
                        table_md = "### Query Results\n\n| " + " | ".join(headers) + " |\n"
                        table_md += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                        
                        # Add rows
                        for row in table_results:
                            row_values = []
                            for header in headers:
                                # Convert values to strings and handle any special formatting
                                value = str(row.get(header, "")).replace("|", "\\|")
                                row_values.append(value)
                            table_md += "| " + " | ".join(row_values) + " |\n"
                        
                        # Send table as markdown
                        await ch.Message(content=table_md).send()
                    else:
                        await ch.Message(content="No results found for your query.").send()
                else:
                    error_text = await response.text()
                    await ch.Message(content=f"Error: Server responded with status code {response.status}. Details: {error_text}").send()
    
    except asyncio.TimeoutError:
        await ch.Message(content="The model is taking too long to respond. Please try again or try a simpler query.").send()
    except aiohttp.ClientConnectorError as e:
        await ch.Message(content=f"Could not connect to the backend server at {url}. Make sure it's running and accessible. Error: {str(e)}").send()
    except Exception as e:
        await ch.Message(content=f"An unexpected error occurred: {str(e)}").send()
        # For debugging
        await ch.Message(content=f"Error details: {type(e).__name__}: {str(e)}").send()


@ch.on_chat_end
async def on_chat_end():
    # Clean up resources or save session data if needed
    pass
