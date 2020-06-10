# frozen_string_literal: true

require "utils/bottles"

describe Utils::Bottles do
  describe "#tag", :needs_macos do
    it "returns :catalina on Catalina" do
      allow(MacOS).to receive(:version).and_return(MacOS::Version.new("10.15"))
      expect(described_class.tag).to eq(:catalina)
    end
  end
end
