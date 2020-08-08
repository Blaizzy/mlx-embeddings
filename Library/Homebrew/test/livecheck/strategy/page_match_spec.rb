# frozen_string_literal: true

require "livecheck/strategy/page_match"

describe Homebrew::Livecheck::Strategy::PageMatch do
  subject(:page_match) { described_class }

  let(:url) { "http://api.github.com/Homebrew/brew/releases/latest" }

  describe "::match?" do
    it "returns true for any URL" do
      expect(page_match.match?(url)).to be true
    end
  end
end
