# typed: false
# frozen_string_literal: true

require "build_environment"

describe BuildEnvironment do
  let(:env) { described_class.new }

  describe "#<<" do
    it "returns itself" do
      expect(env << :foo).to be env
    end
  end

  describe "#merge" do
    it "returns itself" do
      expect(env.merge([])).to be env
    end
  end

  describe "#std?" do
    it "returns true if the environment contains :std" do
      env << :std
      expect(env).to be_std
    end

    it "returns false if the environment does not contain :std" do
      expect(env).not_to be_std
    end
  end

  describe BuildEnvironment::DSL do
    subject(:build_environment_dsl) { double.extend(described_class) }

    context "with a single argument" do
      before do
        build_environment_dsl.instance_eval do
          env :std
        end
      end

      its(:env) { is_expected.to be_std }
    end
  end
end
