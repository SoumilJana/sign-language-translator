export default function Home() {
  return (
    <div className="max-w-4xl mx-auto text-center space-y-10">
      <section className="mt-10">
        <h1 className="text-4xl font-bold">Welcome to SignVision</h1>
        <p className="mt-4 text-lg text-gray-700">
          Bridging communication between sign language and text.
        </p>
      </section>

      <section id="about" className="mt-16">
        <h2 className="text-2xl font-bold">About Us</h2>
        <p className="mt-4 text-gray-700">
          SignVision is a project that enables real-time translation of sign
          language gestures into text and speech. Our mission is to promote
          accessibility and inclusivity with AI-powered tools.
        </p>
      </section>
    </div>
  );
}
