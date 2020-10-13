# typed: false
# frozen_string_literal: true

require "livecheck/strategy/launchpad"

describe Homebrew::Livecheck::Strategy::Launchpad do
  subject(:launchpad) { described_class }

  let(:launchpad_url) { "https://launchpad.net/abc/1.2/1.2.3/+download/def-1.2.3.tar.gz" }
  let(:non_launchpad_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a Launchpad URL" do
      expect(launchpad.match?(launchpad_url)).to be true
    end

    it "returns false if the argument provided is not a Launchpad URL" do
      expect(launchpad.match?(non_launchpad_url)).to be false
    end
  end
end
