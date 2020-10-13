# typed: false
# frozen_string_literal: true

require "livecheck/strategy/xorg"

describe Homebrew::Livecheck::Strategy::Xorg do
  subject(:xorg) { described_class }

  let(:xorg_url) { "https://www.x.org/archive/individual/app/abc-1.2.3.tar.bz2" }
  let(:non_xorg_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is an X.Org URL" do
      expect(xorg.match?(xorg_url)).to be true
    end

    it "returns false if the argument provided is not an X.Org URL" do
      expect(xorg.match?(non_xorg_url)).to be false
    end
  end
end
