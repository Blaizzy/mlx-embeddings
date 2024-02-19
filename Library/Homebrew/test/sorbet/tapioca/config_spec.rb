# frozen_string_literal: true

require "rubygems"
require "yaml"

RSpec.describe "Tapioca Config" do
  let(:config) { YAML.load_file(File.join(__dir__, "../../../sorbet/tapioca/config.yml")) }

  it "only excludes dependencies" do
    exclusions = config.dig("gem", "exclude")
    dependencies = Gem::Specification.all.map(&:name)
    expect(exclusions - dependencies).to be_empty
  end
end
