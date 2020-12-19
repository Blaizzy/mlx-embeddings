# typed: false
# frozen_string_literal: true

require "livecheck/strategy/header_match"

describe Homebrew::Livecheck::Strategy::HeaderMatch do
  subject(:header_match) { described_class }

  let(:url) { "https://www.example.com/" }

  describe "::match?" do
    it "returns true for any URL" do
      expect(header_match.match?(url)).to be true
    end
  end
end
