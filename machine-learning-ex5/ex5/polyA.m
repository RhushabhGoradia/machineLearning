function [apoly] = polyA(a, p)
  
  apoly = zeros(numel(a),p)
  for i = 1:p
    apoly(:, i) = a.^i;
  endfor
end