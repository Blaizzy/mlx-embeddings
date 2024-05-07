# frozen_string_literal: true

require "sbom"

RSpec.describe SBOM, :needs_network do
  describe "#valid?" do
    it "returns true if the SBOM is valid" do
      f = formula do
        url "foo-1.0"
      end

      sbom = described_class.create(f)
      expect(sbom).to be_valid
    end

    it "returns true if the SBOM is valid with dependencies" do
      f = formula do
        url "foo-1.0"

        # some random dependencies to test with
        depends_on "cmake" => :build
        depends_on "beanstalkd"

        uses_from_macos "python" => :build
        uses_from_macos "zlib"
      end

      beanstalkd = formula "beanstalkd" do
        url "one-1.1"
      end

      zlib = formula "zlib" do
        url "two-1.1"
      end

      allow(f).to receive_messages(
        runtime_formula_dependencies: [beanstalkd, zlib],
      )

      sbom = described_class.create(f)
      expect(sbom).to be_valid
    end

    it "returns true if SBOM is valid with patches" do
      f = formula do
        homepage "https://brew.sh"

        url "https://brew.sh/test-0.1.tbz"
        sha256 TEST_SHA256

        patch do
          url "patch_macos"
        end
      end

      sbom = described_class.create(f)
      expect(sbom).to be_valid
    end
  end
end
