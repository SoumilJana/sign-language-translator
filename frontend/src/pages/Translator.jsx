import Webcam from "react-webcam";
import { useState } from "react";

export default function Translator() {
  const [translation, setTranslation] = useState("");

  // Placeholder handler (you will replace with API later)
  const handleFakeTranslate = () => {
    setTranslation((prev) => prev + "A");
  };

  return (
    <div className="grid md:grid-cols-2 gap-6 max-w-6xl mx-auto mt-6">
      {/* Left Column */}
      <div className="space-y-6">
        <div className="border rounded-2xl p-4 shadow bg-white">
          <h2 className="font-bold mb-2">Camera Feed</h2>
          <Webcam className="w-full h-64 rounded-xl border" />
        </div>

        <div className="border rounded-2xl p-4 shadow bg-white">
          <h2 className="font-bold mb-2">Translation Output</h2>
          <div className="min-h-[100px] p-2 bg-gray-100 rounded-md text-left">
            {translation || "Your translation will appear here..."}
          </div>
          <button
            onClick={handleFakeTranslate}
            className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-md"
          >
            Fake Translate
          </button>
        </div>
      </div>

      {/* Right Column */}
      <div className="space-y-6">
        <div className="border rounded-2xl p-4 shadow bg-white">
          <h2 className="font-bold mb-2">Our Work</h2>
          <p className="text-gray-700">
            This section can display recent research, examples, or project
            impact.
          </p>
        </div>

        <div className="border rounded-2xl p-4 shadow bg-white">
          <h2 className="font-bold mb-2">Keyboard Controls</h2>
          <ul className="space-y-2 text-gray-700">
            <li>Complete Sentence: <kbd className="px-2 py-1 bg-gray-200 rounded">Enter</kbd></li>
            <li>Speak: <kbd className="px-2 py-1 bg-gray-200 rounded">Enter</kbd></li>
            <li>Delete Last Character: <kbd className="px-2 py-1 bg-gray-200 rounded">Del/Backspace</kbd></li>
            <li>Delete Last Word: <kbd className="px-2 py-1 bg-gray-200 rounded">Ctrl + Del</kbd></li>
            <li>Quit: <kbd className="px-2 py-1 bg-gray-200 rounded">Q/Esc</kbd></li>
          </ul>
        </div>
      </div>
    </div>
  );
}
