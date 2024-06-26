using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace AskAiCli;

internal static class Program
{
    private const string ContextFilePath = "context.txt";

    private const string ModelPath =
        @"C:\Users\Mimiyo\Documents\_Code\AI\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32";

    private const string SystemPrompt =
        "You are an AI CLI assistant that helps developers find information about terminal commands inside terminal. " +
        "Answer using a direct style. Do not provide more information than requested by user. " +
        "Answer user's question as short as possible.";

    private static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: ask <your question>");
            return;
        }

        var userCommand = args[0].ToLower();
        if (userCommand == "clear" && args.Length == 1)
        {
            UpdateContext(string.Empty);
            Console.WriteLine("Context cleared.");
            return;
        }

        try
        {
            var model = new Model(ModelPath);
            var tokenizer = new Tokenizer(model);
            var userPrompt = string.Join(" ", args);
            var context = GetOrInitializeContext(userPrompt);

            var tokens = tokenizer.Encode(context);
            var generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetSearchOption("past_present_share_buffer", false);
            generatorParams.SetInputSequences(tokens);

            var generator = new Generator(model, generatorParams);
            var result = GenerateResponse(generator, tokenizer);

            UpdateContext(context + result);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
        }
    }

    private static string GetOrInitializeContext(string userPrompt)
    {
        var context = ReadContext();
        if (string.IsNullOrWhiteSpace(context))
        {
            context = $"<|system|>{SystemPrompt}<|end|><|user|>{userPrompt}<|end|><|assistant|>";
        }
        else
        {
            context += $"<|user|>{userPrompt}<|end|><|assistant|>";
        }

        return context;
    }

    private static string GenerateResponse(Generator generator, Tokenizer tokenizer)
    {
        var result = new StringBuilder();
        while (!generator.IsDone())
        {
            generator.ComputeLogits();
            generator.GenerateNextToken();
            var outputTokens = generator.GetSequence(0);
            var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
            var output = tokenizer.Decode(newToken);
            Console.Write(output);
            result.Append(output);
        }

        return result.ToString();
    }

    private static void UpdateContext(string context)
    {
        try
        {
            File.WriteAllText(ContextFilePath, context);
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Failed to update context: {ex.Message}");
        }
    }

    private static string ReadContext()
    {
        try
        {
            return File.Exists(ContextFilePath) ? File.ReadAllText(ContextFilePath) : string.Empty;
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Failed to read context: {ex.Message}");
            return string.Empty;
        }
    }
}