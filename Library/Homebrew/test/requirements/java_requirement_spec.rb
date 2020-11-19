# typed: false
# frozen_string_literal: true

require "requirements/java_requirement"

describe JavaRequirement do
  subject { described_class.new([]) }

  before do
    ENV["JAVA_HOME"] = nil
  end

  describe "#initialize" do
    it "parses '1.8' tag correctly" do
      req = described_class.new(["1.8"])
      expect(req.display_s).to eq("Java = 1.8")
    end

    it "parses '9' tag correctly" do
      req = described_class.new(["9"])
      expect(req.display_s).to eq("Java = 9")
    end

    it "parses '9+' tag correctly" do
      req = described_class.new(["9+"])
      expect(req.display_s).to eq("Java >= 9")
    end

    it "parses '11' tag correctly" do
      req = described_class.new(["11"])
      expect(req.display_s).to eq("Java = 11")
    end

    it "parses bogus tag correctly" do
      req = described_class.new(["bogus1.8"])
      expect(req.display_s).to eq("Java")
    end
  end

  describe "#message" do
    its(:message) { is_expected.to match(/Java is required for this software./) }
  end

  describe "#inspect" do
    subject { described_class.new(%w[1.7+]) }

    its(:inspect) { is_expected.to eq('#<JavaRequirement: version="1.7+" []>') }
  end

  describe "#display_s" do
    context "without specific version" do
      its(:display_s) { is_expected.to eq("Java") }
    end

    context "with version 1.8" do
      subject { described_class.new(%w[1.8]) }

      its(:display_s) { is_expected.to eq("Java = 1.8") }
    end

    context "with version 1.8+" do
      subject { described_class.new(%w[1.8+]) }

      its(:display_s) { is_expected.to eq("Java >= 1.8") }
    end
  end

  describe "#satisfied?" do
    subject(:requirement) { described_class.new(%w[1.8]) }

    if !OS.mac? || MacOS.version < :big_sur
      it "returns false if no `java` executable can be found" do
        allow(File).to receive(:executable?).and_return(false)
        expect(requirement).not_to be_satisfied
      end
    end

    it "returns true if #preferred_java returns a path" do
      allow(requirement).to receive(:preferred_java).and_return(Pathname.new("/usr/bin/java"))
      expect(requirement).to be_satisfied
    end

    context "when #possible_javas contains paths" do
      let(:path) { mktmpdir }
      let(:java) { path/"java" }

      def setup_java_with_version(version)
        IO.write java, <<~SH
          #!/bin/sh
          echo 'java version "#{version}"' 1>&2
        SH
        FileUtils.chmod "+x", java
      end

      before do
        allow(requirement).to receive(:possible_javas).and_return([java])
      end

      context "and 1.7 is required" do
        subject(:requirement) { described_class.new(%w[1.7]) }

        it "returns false if all are lower" do
          setup_java_with_version "1.6.0_5"
          expect(requirement).not_to be_satisfied
        end

        it "returns true if one is equal" do
          setup_java_with_version "1.7.0_5"
          expect(requirement).to be_satisfied
        end

        it "returns false if all are higher" do
          setup_java_with_version "1.8.0_5"
          expect(requirement).not_to be_satisfied
        end
      end

      context "and 1.7+ is required" do
        subject(:requirement) { described_class.new(%w[1.7+]) }

        it "returns false if all are lower" do
          setup_java_with_version "1.6.0_5"
          expect(requirement).not_to be_satisfied
        end

        it "returns true if one is equal" do
          setup_java_with_version "1.7.0_5"
          expect(requirement).to be_satisfied
        end

        it "returns true if one is higher" do
          setup_java_with_version "1.8.0_5"
          expect(requirement).to be_satisfied
        end
      end
    end
  end

  describe "#suggestion" do
    context "without specific version" do
      its(:suggestion) { is_expected.to match(/brew cask install adoptopenjdk/) }
      its(:cask) { is_expected.to eq("adoptopenjdk") }
    end

    context "with version 1.8" do
      subject { described_class.new(%w[1.8]) }

      its(:suggestion) { is_expected.to match(%r{brew cask install homebrew/cask-versions/adoptopenjdk8}) }
      its(:cask) { is_expected.to eq("homebrew/cask-versions/adoptopenjdk8") }
    end

    context "with version 1.8+" do
      subject { described_class.new(%w[1.8+]) }

      its(:suggestion) { is_expected.to match(/brew cask install adoptopenjdk/) }
      its(:cask) { is_expected.to eq("adoptopenjdk") }
    end
  end
end
