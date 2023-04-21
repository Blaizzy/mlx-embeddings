
# frozen_string_literal: true

require "macos_runner_spec"

describe MacOSRunnerSpec do
  let(:spec) { described_class.new(name: "macOS 11-arm64", runner: "11-arm64", cleanup: true) }

  it "has immutable attributes" do
    [:name, :runner, :cleanup].each do |attribute|
      expect(spec.respond_to?("#{attribute}=")).to be(false)
    end
  end

  describe "#to_h" do
    it "returns an object that responds to `#to_json`" do
      expect(spec.to_h.respond_to?(:to_json)).to be(true)
    end
  end
end
