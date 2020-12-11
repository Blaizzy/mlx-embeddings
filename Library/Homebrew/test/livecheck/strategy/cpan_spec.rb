# typed: false
# frozen_string_literal: true

require "livecheck/strategy/cpan"

describe Homebrew::Livecheck::Strategy::Cpan do
  subject(:cpan) { described_class }

  let(:cpan_url) { "https://cpan.metacpan.org/authors/id/M/MI/MIYAGAWA/Carton-v1.0.34.tar.gz" }
  let(:non_cpan_url) { "https://brew.sh/test" }

  describe "::match?" do
    it "returns true if the argument provided is a CPAN URL" do
      expect(cpan.match?(cpan_url)).to be true
    end

    it "returns false if the argument provided is not a CPAN URL" do
      expect(cpan.match?(non_cpan_url)).to be false
    end
  end
end
