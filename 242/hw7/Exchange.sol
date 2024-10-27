// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.7;
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract Exchange {
    mapping (address => uint) private liqPos;
    uint sumLiqPos = 0;
    uint numLiqPos = 0;
    uint K = 0;
    ERC20 public immutable token;

    event LiquidityProvided(uint amountERC20TokenDeposited, uint amountEthDeposited, uint liquidityPositionsIssued);
    event LiquidityWithdrew(uint amountERC20TokenWithdrew, uint amountEthWithdrew, uint liquidityPositionsBurned);
    event SwapForEth(uint amountERC20TokenDeposited, uint amountEthWithdrew);
    event SwapForERC20Token(uint amountERC20TokenWithdrew, uint amountEthDeposited);

    constructor(address _tokenAddress) {
        token = ERC20(_tokenAddress);
    }

    function provideLiquidity(uint _amountERC20Token) public payable returns (uint) {
        require(msg.value > 0 && _amountERC20Token > 0, "Token values must be positive");
        uint preErcBal = getErc20Bal();
        uint preEthBal = getEthBal() - msg.value;
        uint postErcBal = getErc20Bal() + _amountERC20Token;
        uint postEthBal = getEthBal();
        uint liq;

        if (preEthBal == 0 || preErcBal == 0) {
            liq = 100;
        } else {
            require(
                sumLiqPos * _amountERC20Token / postErcBal == sumLiqPos * msg.value / postEthBal,
                "Invalid Ratio"
            );
            liq = sumLiqPos * _amountERC20Token / preErcBal;
        }
        if (!token.transferFrom(msg.sender, address(this), _amountERC20Token)) {
            revert("transferFrom has failed.");
        }
        liqPos[msg.sender] += liq;
        sumLiqPos += liq;
        K = postErcBal * postEthBal;

        emit LiquidityProvided(_amountERC20Token, msg.value, liq);
        return liq;
    }

    function withdrawLiquidity(uint _liquidityPositionsToBurn) public returns (uint, uint) {
        require(_liquidityPositionsToBurn <= liqPos[msg.sender], "You do not have enough liquidity for this action");
        require(_liquidityPositionsToBurn < sumLiqPos, "This action would deplete the liquidity pool");

        uint ethToSend = _liquidityPositionsToBurn * getEthBal() / sumLiqPos;
        uint ercToSend = _liquidityPositionsToBurn * getErc20Bal() / sumLiqPos;
        token.transfer(msg.sender, ercToSend);
        payable(msg.sender).transfer(ethToSend);

        liqPos[msg.sender] -= _liquidityPositionsToBurn;
        sumLiqPos -= _liquidityPositionsToBurn;
        K = getErc20Bal() * getErc20Bal();

        emit LiquidityWithdrew(ercToSend, ethToSend, _liquidityPositionsToBurn);
        return (ercToSend, ethToSend);
    }

    function swapForEth(uint _amountERC20Token) public returns (uint) {
        require(_amountERC20Token > 0, "_amountERC20Token must be positive");
        uint ethAfterSwap = K / (getErc20Bal() + _amountERC20Token);
        require(ethAfterSwap > 0, "Contract does not have enough ETH supply");
        uint ethToSend = getEthBal() - ethAfterSwap;
        if (!token.transferFrom(msg.sender, address(this), _amountERC20Token)) {
            revert("transferFrom has failed.");
        }
        payable(msg.sender).transfer(ethToSend);
        emit SwapForEth(_amountERC20Token, ethToSend);
        return ethToSend;
    }

    function swapForERC20Token() public payable returns (uint) {
        require(msg.value > 0, "msg.value must be positive");
        uint ercAfterSwap = K / (getEthBal()); //do not need to add msg.value here because it is done automatically
        require(ercAfterSwap > 0, "Contract does not have enough ERC supply");
        uint ercToSend = getErc20Bal() - ercAfterSwap;
        token.transfer(msg.sender, ercToSend);
        emit SwapForERC20Token(ercToSend, msg.value);
        return ercToSend;
    }

    function estimateEthToProvide(uint _amountERC20Token) public view returns (uint) {
        require(sumLiqPos != 0, "No liquidity");
        return getEthBal() * _amountERC20Token / getErc20Bal();
    }

    function estimateERC20TokenToProvide(uint _amountEth) public view returns (uint) {
        require(sumLiqPos != 0, "No liquidity");
        return getErc20Bal() * _amountEth / getEthBal();
    }

    function estimateSwapForEth(uint _amountERC20Token) public view returns (uint) {
        require(_amountERC20Token > 0, "_amountERC20Token must be positive");
        uint ethAfterSwap = K / (getErc20Bal() + _amountERC20Token);
        require(ethAfterSwap > 0, "Contract does not have enough ETH supply");
        return getEthBal() - ethAfterSwap;
    }

    function estimateSwapForERC20Token(uint _amountEth) public view returns (uint) {
        require(_amountEth > 0, "_amountEth must be positive");
        uint ercAfterSwap = K / (getEthBal() + _amountEth);
        require(ercAfterSwap > 0, "Contract does not have enough ERC supply");
        return getErc20Bal() - ercAfterSwap;
    }

    function getMyLiquidityPositions() public view returns (uint) {
        return liqPos[msg.sender];
    }

    function getErc20Bal() private view returns (uint256) {
        return token.balanceOf(address(this));
    }

    function getEthBal() private view returns (uint256) {
        return address(this).balance;
    }
}
