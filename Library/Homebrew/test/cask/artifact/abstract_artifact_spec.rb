# typed: false
# frozen_string_literal: true

describe Cask::Artifact::AbstractArtifact, :cask do
  describe ".read_script_arguments" do
    it "accepts a string, and uses it as the executable" do
      arguments = "something"
      stanza = :installer

      expect(described_class.read_script_arguments(arguments, stanza)).to eq(["something", {}])
    end

    it "accepts a hash with an executable" do
      arguments = { executable: "something" }
      stanza = :installer

      expect(described_class.read_script_arguments(arguments, stanza)).to eq(["something", {}])
    end

    it "does not mutate the arguments in place" do
      arguments = { executable: "something", foo: "bar" }
      clone = arguments.dup
      stanza = :installer

      described_class.read_script_arguments(arguments, stanza)

      expect(arguments).to eq(clone)
    end
  end
end
