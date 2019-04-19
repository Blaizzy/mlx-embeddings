# frozen_string_literal: true

require "utils/bottles"

describe Utils::Bottles do
  describe "#tag", :needs_macos do
    it "returns :mavericks on Mavericks" do
      allow(MacOS).to receive(:version).and_return(MacOS::Version.new("10.9"))
      expect(described_class.tag).to eq(:mavericks)
    end
  end
end
