# typed: false
# frozen_string_literal: true

require "livecheck/strategy/hackage"

describe Homebrew::Livecheck::Strategy::Hackage do
  subject(:hackage) { described_class }

  let(:hackage_url) { "https://hackage.haskell.org/package/abc-1.2.3/def-1.2.3.tar.gz" }
  let(:hackage_downloads_url) { "https://downloads.haskell.org/~abc/1.2.3/def-1.2.3-src.tar.xz" }
  let(:non_hackage_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a Hackage URL" do
      expect(hackage.match?(hackage_url)).to be true
      expect(hackage.match?(hackage_downloads_url)).to be true
    end

    it "returns false if the argument provided is not a Hackage URL" do
      expect(hackage.match?(non_hackage_url)).to be false
    end
  end
end
