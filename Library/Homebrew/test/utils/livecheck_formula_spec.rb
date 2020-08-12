# frozen_string_literal: true

require "utils/livecheck_formula"
require "formula_installer"

describe LivecheckFormula do
  describe "init" do
    let(:f) { formula { url "foo-1.0" } }
    let(:options) { FormulaInstaller.new(f).display_options(f) }
    let(:action)  { "#{f.full_name} #{options}".strip }

    it "runs livecheck command for Formula" do
      formatted_response = described_class.init(action)

      expect(formatted_response).not_to be_nil
      expect(formatted_response).to be_a(Hash)
      expect(formatted_response.size).not_to eq(0)
    end
  end

  describe "parse_livecheck_response" do
    it "returns a hash of Formula version data" do
      example_raw_command_response = "aacgain : 7834 ==> 1.8"
      formatted_response = described_class.parse_livecheck_response(example_raw_command_response)

      expect(formatted_response).not_to be_nil
      expect(formatted_response).to be_a(Hash)

      expect(formatted_response).to include(:name)
      expect(formatted_response).to include(:formula_version)
      expect(formatted_response).to include(:livecheck_version)

      expect(formatted_response[:name]).to eq("aacgain")
      expect(formatted_response[:formula_version]).to eq("7834")
      expect(formatted_response[:livecheck_version]).to eq("1.8")
    end
  end
end
