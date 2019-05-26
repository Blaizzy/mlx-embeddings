# frozen_string_literal: true

require "software_spec"

describe SoftwareSpec do
  subject(:spec) { described_class.new }

  describe "#uses_from_macos" do
    before do
      allow(OS).to receive(:linux?).and_return(true)
    end

    it "allows specifying dependencies" do
      spec.uses_from_macos("foo")

      expect(spec.deps.first.name).to eq("foo")
    end

    it "works with tags" do
      spec.uses_from_macos("foo" => :head, :after => :mojave)

      expect(spec.deps.first.name).to eq("foo")
      expect(spec.deps.first.tags).to include(:head)
    end

    it "ignores OS version specifications" do
      spec.uses_from_macos("foo", after: :mojave)
      spec.uses_from_macos("bar" => :head, :after => :mojave)

      expect(spec.deps.first.name).to eq("foo")
      expect(spec.deps.last.name).to eq("bar")
      expect(spec.deps.last.tags).to include(:head)
    end
  end
end
