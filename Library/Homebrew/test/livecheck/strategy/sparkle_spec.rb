# typed: false
# frozen_string_literal: true

require "livecheck/strategy/sparkle"

describe Homebrew::Livecheck::Strategy::Sparkle do
  subject(:sparkle) { described_class }

  let(:url) { "https://www.example.com/example/appcast.xml" }

  describe "::match?" do
    it "returns true for any URL" do
      expect(sparkle.match?(url)).to be true
    end
  end
end
