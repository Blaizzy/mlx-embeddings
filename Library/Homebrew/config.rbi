# typed: strict

module EnvVar
  sig { params(env: String).returns(String) }
  def self.[](env); end
end
